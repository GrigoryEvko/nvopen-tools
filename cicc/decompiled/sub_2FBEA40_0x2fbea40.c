// Function: sub_2FBEA40
// Address: 0x2fbea40
//
__int64 __fastcall sub_2FBEA40(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2FACF50((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 26;
    *(_QWORD *)v1 = "Merge disjoint stack slots";
    *(_QWORD *)(v1 + 16) = "stack-coloring";
    *(_QWORD *)(v1 + 32) = &unk_5025D0C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 14;
    *(_QWORD *)(v1 + 48) = sub_2FBE930;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
