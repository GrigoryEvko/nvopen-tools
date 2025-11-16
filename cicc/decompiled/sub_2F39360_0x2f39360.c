// Function: sub_2F39360
// Address: 0x2f39360
//
__int64 __fastcall sub_2F39360(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 39;
    *(_QWORD *)v1 = "Post RA top-down list latency scheduler";
    *(_QWORD *)(v1 + 16) = "post-RA-sched";
    *(_QWORD *)(v1 + 32) = &unk_5022FAC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 13;
    *(_QWORD *)(v1 + 48) = sub_2F39250;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
