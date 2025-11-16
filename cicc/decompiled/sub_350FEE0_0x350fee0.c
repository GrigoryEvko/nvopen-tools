// Function: sub_350FEE0
// Address: 0x350fee0
//
__int64 __fastcall sub_350FEE0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2E44030((__int64)rwlock);
  sub_2E399F0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 27;
    *(_QWORD *)v1 = "Basic Block Placement Stats";
    *(_QWORD *)(v1 + 16) = "block-placement-stats";
    *(_QWORD *)(v1 + 32) = &unk_503BDBC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 21;
    *(_QWORD *)(v1 + 48) = sub_3517680;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
