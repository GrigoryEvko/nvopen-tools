// Function: sub_2DE1AB0
// Address: 0x2de1ab0
//
__int64 __fastcall sub_2DE1AB0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  sub_D4AA90((__int64)rwlock);
  sub_D9A960((__int64)rwlock);
  sub_1049990((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 23;
    *(_QWORD *)v1 = "Hardware Loop Insertion";
    *(_QWORD *)(v1 + 16) = "hardware-loops";
    *(_QWORD *)(v1 + 32) = &unk_501E20C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 14;
    *(_QWORD *)(v1 + 48) = sub_2DE20A0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
