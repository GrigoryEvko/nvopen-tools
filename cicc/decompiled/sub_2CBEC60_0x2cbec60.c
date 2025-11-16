// Function: sub_2CBEC60
// Address: 0x2cbec60
//
__int64 __fastcall sub_2CBEC60(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  sub_D4AA90((__int64)rwlock);
  sub_11CDF60((__int64)rwlock);
  sub_F67EE0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 17;
    *(_QWORD *)v1 = "Index Split Loops";
    *(_QWORD *)(v1 + 16) = "loop-index-split";
    *(_QWORD *)(v1 + 32) = &unk_501358C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 16;
    *(_QWORD *)(v1 + 48) = sub_2CBFF20;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
