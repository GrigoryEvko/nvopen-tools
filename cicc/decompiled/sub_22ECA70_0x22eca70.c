// Function: sub_22ECA70
// Address: 0x22eca70
//
__int64 __fastcall sub_22ECA70(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 21;
    *(_QWORD *)v1 = "Safepoint IR Verifier";
    *(_QWORD *)(v1 + 16) = "verify-safepoint-ir";
    *(_QWORD *)(v1 + 32) = &unk_4FDC19C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 19;
    *(_QWORD *)(v1 + 48) = sub_22ED220;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
