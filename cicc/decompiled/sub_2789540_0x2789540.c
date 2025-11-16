// Function: sub_2789540
// Address: 0x2789540
//
__int64 __fastcall sub_2789540(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_CFB980((__int64)rwlock);
  sub_102CFA0((__int64)rwlock);
  sub_103BB40((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_97FFF0((__int64)rwlock);
  sub_CF6DB0((__int64)rwlock);
  sub_D1D8C0((__int64)rwlock);
  sub_1049990((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 22;
    *(_QWORD *)v1 = "Global Value Numbering";
    *(_QWORD *)(v1 + 16) = "gvn";
    *(_QWORD *)(v1 + 32) = &unk_4FFB38C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 3;
    *(_QWORD *)(v1 + 48) = sub_278C3F0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
