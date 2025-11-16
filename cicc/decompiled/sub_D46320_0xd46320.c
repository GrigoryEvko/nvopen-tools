// Function: sub_D46320
// Address: 0xd46320
//
__int64 __fastcall sub_D46320(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 24;
    *(_QWORD *)v1 = "Natural Loop Information";
    *(_QWORD *)(v1 + 16) = "loops";
    *(_QWORD *)(v1 + 24) = 5;
    *(_QWORD *)(v1 + 32) = &unk_4F875EC;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_D4ACA0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
