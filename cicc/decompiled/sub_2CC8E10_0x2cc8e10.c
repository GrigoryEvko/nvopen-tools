// Function: sub_2CC8E10
// Address: 0x2cc8e10
//
__int64 __fastcall sub_2CC8E10(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 22;
    *(_QWORD *)v1 = "Lower Aggregate Copies";
    *(_QWORD *)(v1 + 16) = "lower-aggr-copies";
    *(_QWORD *)(v1 + 24) = 17;
    *(_QWORD *)(v1 + 32) = &unk_50139EC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_2CC96B0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
