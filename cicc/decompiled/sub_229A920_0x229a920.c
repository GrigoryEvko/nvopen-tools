// Function: sub_229A920
// Address: 0x229a920
//
__int64 __fastcall sub_229A920(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 72;
    *(_QWORD *)v1 = "Print dominance tree of function to 'dot' file (with no function bodies)";
    *(_QWORD *)(v1 + 16) = "dot-dom-only";
    *(_QWORD *)(v1 + 32) = &unk_4FDB5FC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 12;
    *(_QWORD *)(v1 + 48) = sub_229C5C0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
