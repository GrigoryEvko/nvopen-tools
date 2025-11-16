// Function: sub_2D4FFC0
// Address: 0x2d4ffc0
//
__int64 __fastcall sub_2D4FFC0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 48;
    *(_QWORD *)v1 = "Reads and parses a basic block sections profile.";
    *(_QWORD *)(v1 + 16) = "bbsections-profile-reader";
    *(_QWORD *)(v1 + 24) = 25;
    *(_QWORD *)(v1 + 32) = &unk_501695C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_2D50DC0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
