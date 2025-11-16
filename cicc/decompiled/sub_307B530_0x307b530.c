// Function: sub_307B530
// Address: 0x307b530
//
__int64 __fastcall sub_307B530(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2EA61A0((__int64)rwlock);
  sub_2E6D3E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 41;
    *(_QWORD *)v1 = "Register pressure analysis on Machine IRs";
    *(_QWORD *)(v1 + 16) = "machine-rpa";
    *(_QWORD *)(v1 + 24) = 11;
    *(_QWORD *)(v1 + 32) = &unk_502D274;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_307BF30;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
