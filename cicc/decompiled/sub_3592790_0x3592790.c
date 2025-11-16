// Function: sub_3592790
// Address: 0x3592790
//
__int64 __fastcall sub_3592790(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 32;
    *(_QWORD *)v1 = "Register Allocation Scoring Pass";
    *(_QWORD *)(v1 + 16) = "regallocscoringpass";
    *(_QWORD *)(v1 + 24) = 19;
    *(_QWORD *)(v1 + 32) = &unk_503F80C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_3595D40;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
