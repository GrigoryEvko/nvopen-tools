// Function: sub_2C761C0
// Address: 0x2c761c0
//
__int64 __fastcall sub_2C761C0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 16;
    *(_QWORD *)v1 = "NVVM IR Verifier";
    *(_QWORD *)(v1 + 16) = "nvvm-verify";
    *(_QWORD *)(v1 + 32) = &unk_5011094;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 11;
    *(_QWORD *)(v1 + 48) = sub_2C78D10;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
