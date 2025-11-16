// Function: sub_1055CB0
// Address: 0x1055cb0
//
__int64 __fastcall sub_1055CB0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  sub_11FC600(rwlock);
  sub_DFEA20((__int64)rwlock);
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 19;
    *(_QWORD *)v1 = "Uniformity Analysis";
    *(_QWORD *)(v1 + 16) = "uniformity";
    *(_QWORD *)(v1 + 24) = 10;
    *(_QWORD *)(v1 + 32) = &unk_4F8FC84;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_1056660;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
