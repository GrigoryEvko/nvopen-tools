// Function: sub_2DC1BE0
// Address: 0x2dc1be0
//
__int64 __fastcall sub_2DC1BE0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_97FFF0((__int64)rwlock);
  sub_DFEA20((__int64)rwlock);
  sub_1027850((__int64)rwlock);
  sub_D84940((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 30;
    *(_QWORD *)v1 = "Expand memcmp() to load/stores";
    *(_QWORD *)(v1 + 16) = "expand-memcmp";
    *(_QWORD *)(v1 + 32) = &unk_501D3CC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 13;
    *(_QWORD *)(v1 + 48) = sub_2DC2E60;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
