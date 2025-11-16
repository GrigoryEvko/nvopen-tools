// Function: sub_36CDB40
// Address: 0x36cdb40
//
__int64 __fastcall sub_36CDB40(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 48;
    *(_QWORD *)v1 = "NVPTX Address space based Alias Analysis Wrapper";
    *(_QWORD *)(v1 + 16) = "nvptx-aa-wrapper";
    *(_QWORD *)(v1 + 24) = 16;
    *(_QWORD *)(v1 + 32) = &unk_5040918;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_36CDE10;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
