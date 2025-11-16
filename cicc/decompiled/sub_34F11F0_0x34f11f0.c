// Function: sub_34F11F0
// Address: 0x34f11f0
//
__int64 __fastcall sub_34F11F0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_CF6DB0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 20;
    *(_QWORD *)v1 = "Implicit null checks";
    *(_QWORD *)(v1 + 16) = "implicit-null-checks";
    *(_QWORD *)(v1 + 32) = &unk_503BAEC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 20;
    *(_QWORD *)(v1 + 48) = sub_34F18F0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
