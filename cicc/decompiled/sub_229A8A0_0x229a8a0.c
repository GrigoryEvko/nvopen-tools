// Function: sub_229A8A0
// Address: 0x229a8a0
//
__int64 __fastcall sub_229A8A0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 46;
    *(_QWORD *)v1 = "Print dominance tree of function to 'dot' file";
    *(_QWORD *)(v1 + 16) = "dot-dom";
    *(_QWORD *)(v1 + 32) = &unk_4FDB604;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 7;
    *(_QWORD *)(v1 + 48) = sub_229C3F0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
