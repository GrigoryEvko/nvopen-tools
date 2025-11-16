// Function: sub_29CD190
// Address: 0x29cd190
//
__int64 __fastcall sub_29CD190(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 74;
    *(_QWORD *)v1 = "Instrument function entry/exit with calls to e.g. mcount() (post inlining)";
    *(_QWORD *)(v1 + 16) = "post-inline-ee-instrument";
    *(_QWORD *)(v1 + 32) = &unk_500900C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 25;
    *(_QWORD *)(v1 + 48) = sub_29CE6B0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
