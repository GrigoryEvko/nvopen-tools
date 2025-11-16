// Function: sub_2DC8BE0
// Address: 0x2dc8be0
//
__int64 __fastcall sub_2DC8BE0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_DFEA20((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 27;
    *(_QWORD *)v1 = "Expand reduction intrinsics";
    *(_QWORD *)(v1 + 16) = "expand-reductions";
    *(_QWORD *)(v1 + 32) = &unk_501D674;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 17;
    *(_QWORD *)(v1 + 48) = sub_2DC8D80;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
