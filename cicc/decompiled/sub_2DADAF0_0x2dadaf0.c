// Function: sub_2DADAF0
// Address: 0x2dadaf0
//
__int64 __fastcall sub_2DADAF0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 17;
    *(_QWORD *)v1 = "Detect Dead Lanes";
    *(_QWORD *)(v1 + 16) = "detect-dead-lanes";
    *(_QWORD *)(v1 + 32) = &unk_501CF54;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 17;
    *(_QWORD *)(v1 + 48) = sub_2DAD9E0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
