// Function: sub_36CCB30
// Address: 0x36ccb30
//
__int64 __fastcall sub_36CCB30(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 61;
    *(_QWORD *)v1 = "Find good alignment for statically sized global/shared arrays";
    *(_QWORD *)(v1 + 16) = "nvptx-set-global-array-alignment";
    *(_QWORD *)(v1 + 32) = &unk_5040830;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 32;
    *(_QWORD *)(v1 + 48) = sub_36CCA40;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
