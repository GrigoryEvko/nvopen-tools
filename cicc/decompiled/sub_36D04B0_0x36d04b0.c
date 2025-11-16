// Function: sub_36D04B0
// Address: 0x36d04b0
//
__int64 __fastcall sub_36D04B0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 33;
    *(_QWORD *)v1 = "Assign valid PTX names to globals";
    *(_QWORD *)(v1 + 16) = "nvptx-assign-valid-global-names";
    *(_QWORD *)(v1 + 32) = &unk_5040BD4;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 31;
    *(_QWORD *)(v1 + 48) = sub_36D0570;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
