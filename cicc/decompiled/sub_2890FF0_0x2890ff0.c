// Function: sub_2890FF0
// Address: 0x2890ff0
//
__int64 __fastcall sub_2890FF0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 42;
    *(_QWORD *)v1 = "Lower atomic intrinsics to non-atomic form";
    *(_QWORD *)(v1 + 16) = "loweratomic";
    *(_QWORD *)(v1 + 32) = &unk_500398C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 11;
    *(_QWORD *)(v1 + 48) = sub_2891520;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
