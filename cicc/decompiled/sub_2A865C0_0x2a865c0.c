// Function: sub_2A865C0
// Address: 0x2a865c0
//
__int64 __fastcall sub_2A865C0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  sub_D4AA90((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 51;
    *(_QWORD *)v1 = "Fixup each natural loop to have a single exit block";
    *(_QWORD *)(v1 + 16) = "unify-loop-exits";
    *(_QWORD *)(v1 + 32) = &unk_500C10C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 16;
    *(_QWORD *)(v1 + 48) = sub_2A86CA0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
