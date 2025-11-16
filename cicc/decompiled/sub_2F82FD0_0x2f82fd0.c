// Function: sub_2F82FD0
// Address: 0x2f82fd0
//
__int64 __fastcall sub_2F82FD0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2FEF6D0(rwlock);
  sub_B1A2E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 31;
    *(_QWORD *)v1 = "Safe Stack instrumentation pass";
    *(_QWORD *)(v1 + 16) = "safe-stack";
    *(_QWORD *)(v1 + 32) = &unk_5024F70;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 10;
    *(_QWORD *)(v1 + 48) = sub_2F84140;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
