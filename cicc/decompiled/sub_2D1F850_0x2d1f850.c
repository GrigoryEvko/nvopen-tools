// Function: sub_2D1F850
// Address: 0x2d1f850
//
__int64 __fastcall sub_2D1F850(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 35;
    *(_QWORD *)v1 = "Check CNP launch calls for legality";
    *(_QWORD *)(v1 + 16) = "cnp-launch-check";
    *(_QWORD *)(v1 + 24) = 16;
    *(_QWORD *)(v1 + 32) = &unk_50164AC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_2D1F960;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
