// Function: sub_272D6B0
// Address: 0x272d6b0
//
__int64 __fastcall sub_272D6B0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_FDC5A0((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_D84940((__int64)rwlock);
  sub_DFEA20((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 17;
    *(_QWORD *)v1 = "Constant Hoisting";
    *(_QWORD *)(v1 + 16) = "consthoist";
    *(_QWORD *)(v1 + 32) = &unk_4FF9DAC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 10;
    *(_QWORD *)(v1 + 48) = sub_2730210;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
