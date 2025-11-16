// Function: sub_25ADFF0
// Address: 0x25adff0
//
__int64 __fastcall sub_25ADFF0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 53;
    *(_QWORD *)v1 = "Dead Argument Hacking (BUGPOINT USE ONLY; DO NOT USE)";
    *(_QWORD *)(v1 + 16) = "deadarghaX0r";
    *(_QWORD *)(v1 + 32) = &unk_4FEFCEC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 12;
    *(_QWORD *)(v1 + 48) = sub_25ADE90;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
