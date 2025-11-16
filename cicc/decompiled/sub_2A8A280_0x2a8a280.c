// Function: sub_2A8A280
// Address: 0x2a8a280
//
__int64 __fastcall sub_2A8A280(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_DF4610((__int64)rwlock);
  sub_CFB980((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_CF6DB0((__int64)rwlock);
  sub_D1D8C0((__int64)rwlock);
  sub_DFEA20((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 37;
    *(_QWORD *)v1 = "Vectorize load and store instructions";
    *(_QWORD *)(v1 + 16) = "load-store-vectorizer";
    *(_QWORD *)(v1 + 32) = &unk_500C1EC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 21;
    *(_QWORD *)(v1 + 48) = sub_2A8DA30;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
