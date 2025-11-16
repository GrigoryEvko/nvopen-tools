// Function: sub_F66B70
// Address: 0xf66b70
//
__int64 __fastcall sub_F66B70(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_CFB980((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_D4AA90((__int64)rwlock);
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 26;
    *(_QWORD *)v1 = "Canonicalize natural loops";
    *(_QWORD *)(v1 + 16) = "loop-simplify";
    *(_QWORD *)(v1 + 32) = &unk_4F8C12C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 13;
    *(_QWORD *)(v1 + 48) = sub_F67F60;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
