// Function: sub_229A7A0
// Address: 0x229a7a0
//
__int64 __fastcall sub_229A7A0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 35;
    *(_QWORD *)v1 = "View postdominance tree of function";
    *(_QWORD *)(v1 + 16) = "view-postdom";
    *(_QWORD *)(v1 + 32) = &unk_4FDB614;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 12;
    *(_QWORD *)(v1 + 48) = sub_229C050;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
