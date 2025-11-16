// Function: sub_2DD31B0
// Address: 0x2dd31b0
//
__int64 __fastcall sub_2DD31B0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 22;
    *(_QWORD *)v1 = "Merge global variables";
    *(_QWORD *)(v1 + 16) = "global-merge";
    *(_QWORD *)(v1 + 32) = &unk_501DA30;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 12;
    *(_QWORD *)(v1 + 48) = sub_2DD6C10;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
