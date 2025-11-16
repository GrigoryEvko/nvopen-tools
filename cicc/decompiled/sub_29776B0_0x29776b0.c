// Function: sub_29776B0
// Address: 0x29776b0
//
__int64 __fastcall sub_29776B0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_D4AA90((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_CF6DB0((__int64)rwlock);
  sub_FCE0C0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 12;
    *(_QWORD *)v1 = "Code sinking";
    *(_QWORD *)(v1 + 16) = "sink";
    *(_QWORD *)(v1 + 32) = &unk_5006DEC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 4;
    *(_QWORD *)(v1 + 48) = sub_2977CE0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
