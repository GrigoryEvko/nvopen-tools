// Function: sub_2D21A80
// Address: 0x2d21a80
//
__int64 __fastcall sub_2D21A80(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_31C5560(rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 34;
    *(_QWORD *)v1 = "early NVVM specific catchall phase";
    *(_QWORD *)(v1 + 16) = "nvvm-pretreat";
    *(_QWORD *)(v1 + 24) = 13;
    *(_QWORD *)(v1 + 32) = &unk_4CE0069;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_2D21C00;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
