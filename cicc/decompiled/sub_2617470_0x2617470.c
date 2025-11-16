// Function: sub_2617470
// Address: 0x2617470
//
__int64 __fastcall sub_2617470(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_F423A0((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_D4AA90((__int64)rwlock);
  sub_F67EE0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 32;
    *(_QWORD *)v1 = "Extract loops into new functions";
    *(_QWORD *)(v1 + 16) = "loop-extract";
    *(_QWORD *)(v1 + 32) = &unk_4FF2A54;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 12;
    *(_QWORD *)(v1 + 48) = sub_2618310;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
