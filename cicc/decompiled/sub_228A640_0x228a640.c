// Function: sub_228A640
// Address: 0x228a640
//
__int64 __fastcall sub_228A640(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_D4AA90((__int64)rwlock);
  sub_D9A960((__int64)rwlock);
  sub_CF6DB0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 19;
    *(_QWORD *)v1 = "Dependence Analysis";
    *(_QWORD *)(v1 + 16) = "da";
    *(_QWORD *)(v1 + 24) = 2;
    *(_QWORD *)(v1 + 32) = &unk_4FDB348;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_228CBB0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
