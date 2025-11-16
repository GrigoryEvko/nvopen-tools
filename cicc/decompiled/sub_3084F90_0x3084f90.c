// Function: sub_3084F90
// Address: 0x3084f90
//
__int64 __fastcall sub_3084F90(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_D9A960((__int64)rwlock);
  sub_DFEA20((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 18;
    *(_QWORD *)v1 = "Ldg Transformation";
    *(_QWORD *)(v1 + 16) = "ldgxform";
    *(_QWORD *)(v1 + 24) = 8;
    *(_QWORD *)(v1 + 32) = &unk_502D430;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_3085670;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
