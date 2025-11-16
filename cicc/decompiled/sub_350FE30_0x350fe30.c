// Function: sub_350FE30
// Address: 0x350fe30
//
__int64 __fastcall sub_350FE30(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2E44030((__int64)rwlock);
  sub_2E399F0((__int64)rwlock);
  sub_2EB3F30((__int64)rwlock);
  sub_2EA61A0((__int64)rwlock);
  sub_D84940((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 40;
    *(_QWORD *)v1 = "Branch Probability Basic Block Placement";
    *(_QWORD *)(v1 + 16) = "block-placement";
    *(_QWORD *)(v1 + 32) = &unk_503BDC4;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 15;
    *(_QWORD *)(v1 + 48) = sub_35172E0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
