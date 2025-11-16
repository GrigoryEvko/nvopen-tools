// Function: sub_229A6A0
// Address: 0x229a6a0
//
__int64 __fastcall sub_229A6A0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 31;
    *(_QWORD *)v1 = "View dominance tree of function";
    *(_QWORD *)(v1 + 16) = "view-dom";
    *(_QWORD *)(v1 + 32) = &unk_4FDB624;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 8;
    *(_QWORD *)(v1 + 48) = sub_229BCB0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
