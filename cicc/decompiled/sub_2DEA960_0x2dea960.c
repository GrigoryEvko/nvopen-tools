// Function: sub_2DEA960
// Address: 0x2dea960
//
__int64 __fastcall sub_2DEA960(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  sub_103BB40((__int64)rwlock);
  sub_DFEA20((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 72;
    *(_QWORD *)v1 = "Combine interleaved loads into wide loads and shufflevector instructions";
    *(_QWORD *)(v1 + 16) = "interleaved-load-combine";
    *(_QWORD *)(v1 + 32) = &unk_501E82C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 24;
    *(_QWORD *)(v1 + 48) = sub_2DEC730;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
