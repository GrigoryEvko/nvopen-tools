// Function: sub_2ED1220
// Address: 0x2ed1220
//
__int64 __fastcall sub_2ED1220(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_D84940((__int64)rwlock);
  sub_2E44030((__int64)rwlock);
  sub_2E6D3E0((__int64)rwlock);
  sub_2E5ED70((__int64)rwlock);
  sub_CF6DB0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 20;
    *(_QWORD *)v1 = "Machine code sinking";
    *(_QWORD *)(v1 + 16) = "machine-sink";
    *(_QWORD *)(v1 + 32) = &unk_5021D2C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 12;
    *(_QWORD *)(v1 + 48) = sub_2ED51F0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
