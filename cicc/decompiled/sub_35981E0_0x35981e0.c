// Function: sub_35981E0
// Address: 0x35981e0
//
__int64 __fastcall sub_35981E0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2EA61A0((__int64)rwlock);
  sub_2E10620((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 25;
    *(_QWORD *)v1 = "Modulo Schedule test pass";
    *(_QWORD *)(v1 + 16) = "modulo-schedule-test";
    *(_QWORD *)(v1 + 32) = &unk_503FC04;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 20;
    *(_QWORD *)(v1 + 48) = sub_359AC40;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
