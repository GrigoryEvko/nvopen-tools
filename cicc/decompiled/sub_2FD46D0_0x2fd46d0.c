// Function: sub_2FD46D0
// Address: 0x2fd46d0
//
__int64 __fastcall sub_2FD46D0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 22;
    *(_QWORD *)v1 = "Early Tail Duplication";
    *(_QWORD *)(v1 + 16) = "early-tailduplication";
    *(_QWORD *)(v1 + 32) = &unk_5026410;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 21;
    *(_QWORD *)(v1 + 48) = sub_2FD52C0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
