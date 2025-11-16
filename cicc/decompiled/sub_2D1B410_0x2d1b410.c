// Function: sub_2D1B410
// Address: 0x2d1b410
//
__int64 __fastcall sub_2D1B410(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  sub_CF6DB0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 12;
    *(_QWORD *)v1 = "Code sinking";
    *(_QWORD *)(v1 + 16) = "sink2";
    *(_QWORD *)(v1 + 32) = &unk_5016214;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 5;
    *(_QWORD *)(v1 + 48) = sub_2D1BC50;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
