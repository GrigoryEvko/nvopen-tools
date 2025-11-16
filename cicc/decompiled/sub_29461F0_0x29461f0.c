// Function: sub_29461F0
// Address: 0x29461f0
//
__int64 __fastcall sub_29461F0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_DFEA20((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 46;
    *(_QWORD *)v1 = "Scalarize unsupported masked memory intrinsics";
    *(_QWORD *)(v1 + 16) = "scalarize-masked-mem-intrin";
    *(_QWORD *)(v1 + 32) = &unk_500571C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 27;
    *(_QWORD *)(v1 + 48) = sub_2946610;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
