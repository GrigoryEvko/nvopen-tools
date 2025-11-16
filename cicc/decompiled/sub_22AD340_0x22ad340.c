// Function: sub_22AD340
// Address: 0x22ad340
//
__int64 __fastcall sub_22AD340(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 24;
    *(_QWORD *)v1 = "ir-similarity-identifier";
    *(_QWORD *)(v1 + 16) = "ir-similarity-identifier";
    *(_QWORD *)(v1 + 24) = 24;
    *(_QWORD *)(v1 + 32) = &unk_4FDB948;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_22AFAB0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
