// Function: sub_34BA6A0
// Address: 0x34ba6a0
//
__int64 __fastcall sub_34BA6A0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2D50D40((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 63;
    *(_QWORD *)v1 = "Applies path clonings for the -basic-block-sections=list option";
    *(_QWORD *)(v1 + 16) = "bb-path-cloning";
    *(_QWORD *)(v1 + 24) = 15;
    *(_QWORD *)(v1 + 32) = &unk_503A62C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_34BA850;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
