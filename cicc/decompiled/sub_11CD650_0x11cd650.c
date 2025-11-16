// Function: sub_11CD650
// Address: 0x11cd650
//
__int64 __fastcall sub_11CD650(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  sub_D4AA90((__int64)rwlock);
  sub_D57FB0((__int64)rwlock);
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 25;
    *(_QWORD *)v1 = "Loop-Closed SSA Form Pass";
    *(_QWORD *)(v1 + 16) = "lcssa";
    *(_QWORD *)(v1 + 32) = &unk_4F90E2C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 5;
    *(_QWORD *)(v1 + 48) = sub_11CDFE0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
