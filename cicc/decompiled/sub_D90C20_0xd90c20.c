// Function: sub_D90C20
// Address: 0xd90c20
//
__int64 __fastcall sub_D90C20(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_CFB980((__int64)rwlock);
  sub_D4AA90((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_97FFF0((__int64)rwlock);
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 25;
    *(_QWORD *)v1 = "Scalar Evolution Analysis";
    *(_QWORD *)(v1 + 16) = "scalar-evolution";
    *(_QWORD *)(v1 + 24) = 16;
    *(_QWORD *)(v1 + 32) = &unk_4F881C8;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_D9AAD0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
