// Function: sub_2DCFD50
// Address: 0x2dcfd50
//
__int64 __fastcall sub_2DCFD50(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 40;
    *(_QWORD *)v1 = "Create Garbage Collector Module Metadata";
    *(_QWORD *)(v1 + 16) = "collector-metadata";
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 32) = &unk_501DA08;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_2DD0C70;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
