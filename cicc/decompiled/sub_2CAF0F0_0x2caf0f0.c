// Function: sub_2CAF0F0
// Address: 0x2caf0f0
//
__int64 __fastcall sub_2CAF0F0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_104C240((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 23;
    *(_QWORD *)v1 = "NVVM Peephole Optimizer";
    *(_QWORD *)(v1 + 16) = "nvvm-peephole-optimizer";
    *(_QWORD *)(v1 + 32) = &unk_5012E8C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 23;
    *(_QWORD *)(v1 + 48) = sub_2CB2280;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
