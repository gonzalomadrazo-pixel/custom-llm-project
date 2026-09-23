// Run with Node; tests the exact math script delivered in the standalone HTML.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const crypto = require('node:crypto');
const html = fs.readFileSync(path.join(__dirname, 'embedding-viewer.html'), 'utf8');
for (const script of html.matchAll(/<script id="(?:math|app)">([\s\S]*?)<\/script>/g)) new vm.Script(script[1]);
const math = html.match(/<script id="math">([\s\S]*?)<\/script>/)[1];
const bundle = JSON.parse(html.match(/<script id="model-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const context = {};
vm.createContext(context);
vm.runInContext(math + ';this.api={dot,cosine,nearest,pca,validateCheckpoint};',context);
const {dot,cosine,nearest,pca,validateCheckpoint} = context.api;
const near = (a,b,tol=1e-9) => assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const checkpointBytes = fs.readFileSync(path.join(__dirname,'examples/reference/checkpoint.json'));
const checkpoint = JSON.parse(checkpointBytes);
assert.deepEqual(bundle.after,checkpoint.weights.wte);
assert.equal(bundle.sha256,crypto.createHash('sha256').update(checkpointBytes).digest('hex'));
assert.equal(bundle.steps,checkpoint.completed_steps);
assert.deepEqual(bundle.before,checkpoint.initial_embeddings);
assert.equal(bundle.after[0].length,64);
const probe=JSON.parse(fs.readFileSync(path.join(__dirname,'examples/reference/inspection.json')));
assert.deepEqual(bundle.before[probe.token_id],probe.embedding_before);
assert.deepEqual(bundle.after[probe.token_id],probe.embedding_after);
near(cosine([1,0],[2,0]),1);near(cosine([1,0],[-2,0]),-1);near(cosine([1,0],[0,2]),0);
assert.equal(cosine([0,0],[1,0]),null);
const rows=[...bundle.before,...bundle.after],fit=pca(rows);
assert.ok(fit.retained>0&&fit.retained<=1);
for(let i=0;i<3;i++){
  near(dot(fit.basis[i],fit.basis[i]),1);
  for(let j=0;j<i;j++)near(dot(fit.basis[i],fit.basis[j]),0);
  const eigenProduct=fit.cov.map(row=>dot(row,fit.basis[i]));
  eigenProduct.forEach((x,j)=>near(x,fit.eigenvalues[i]*fit.basis[i][j]));
}
// Same basis and center for both states: projected displacement equals projected delta.
bundle.before.forEach((before,i)=>{
  const after=bundle.after[i],delta=after.map((x,j)=>x-before[j]);
  fit.basis.forEach((axis,k)=>near(fit.project(after)[k]-fit.project(before)[k],dot(delta,axis)));
});
const projected=rows.map(fit.project),total=fit.eigenvalues.reduce((a,b)=>a+b,0);
const projectedVariance=projected.reduce((s,row)=>s+dot(row,row),0)/(rows.length-1);
near(projectedVariance/total,fit.retained);
near(pca([[1,0,0],[-1,0,0],[0,2,0],[0,-2,0]]).retained,1);
near(pca([[0,0],[0,0]]).retained,0);
near(pca([[1],[2],[3]]).retained,1);
assert.ok(pca([[1],[2],[3]]).project([2]).every(Number.isFinite));
assert.equal(nearest([[1,0],[1,1],[-1,0],[0,1]],0)[0].i,1);
const imported=validateCheckpoint(checkpoint);
assert.equal(JSON.stringify(imported.before),JSON.stringify(bundle.before));
assert.ok(imported.tokens.includes('customer'));
assert.equal(imported.tokens.length,bundle.after.length);
const invalid=[{}, {...checkpoint,completed_steps:-1}, {...checkpoint,vocabulary:['a','a']}, {...checkpoint,weights:{wte:[[1,2],[3]]}}, {...checkpoint,weights:{wte:[[Infinity],[0]]}}, {...checkpoint,weights:{wte:[[1e10],[0]]}}];
invalid.push({...checkpoint,initial_embeddings:[[0,1]]});
for(const input of invalid)assert.throws(()=>validateCheckpoint(input));
// All local resources: direct file opening should not require a network.
assert.equal(/<script[^>]+src=|<link[^>]+href=|fetch\(|XMLHttpRequest|WebSocket/.test(html),false);
console.log(`PASS: bundled checkpoint/probe, cosine, PCA eigenvectors/variance, shared basis, zero/1D data, checkpoint validation, offline resources.`);
console.log(`Pooled PCA retains ${(fit.retained*100).toFixed(4)}% variance; customer neighbors after training:`,nearest(bundle.after,bundle.tokens.indexOf('customer')).map(x=>[bundle.tokens[x.i],x.score]));
