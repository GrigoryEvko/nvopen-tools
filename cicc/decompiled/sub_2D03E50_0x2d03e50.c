// Function: sub_2D03E50
// Address: 0x2d03e50
//
__int64 __fastcall sub_2D03E50(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_D4AA90((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_FDC5A0((__int64)rwlock);
  sub_DFEA20((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 37;
    *(_QWORD *)v1 = "Register Rematerialization on NVVM IR";
    *(_QWORD *)(v1 + 16) = "remat";
    *(_QWORD *)(v1 + 24) = 5;
    *(_QWORD *)(v1 + 32) = &unk_5014E4C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_2D05A60;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
