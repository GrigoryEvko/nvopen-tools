// Function: sub_2CDD6D0
// Address: 0x2cdd6d0
//
__int64 __fastcall sub_2CDD6D0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  sub_CFB980((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 16;
    *(_QWORD *)v1 = "Memory Space Opt";
    *(_QWORD *)(v1 + 16) = "memory-space-opt-pass";
    *(_QWORD *)(v1 + 32) = &unk_50142AC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 21;
    *(_QWORD *)(v1 + 48) = sub_2CDFF20;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
