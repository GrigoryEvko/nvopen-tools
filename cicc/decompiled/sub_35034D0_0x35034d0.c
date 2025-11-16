// Function: sub_35034D0
// Address: 0x35034d0
//
__int64 __fastcall sub_35034D0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2E44030((__int64)rwlock);
  sub_2EA61A0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 37;
    *(_QWORD *)v1 = "Lazy Machine Block Frequency Analysis";
    *(_QWORD *)(v1 + 16) = "lazy-machine-block-freq";
    *(_QWORD *)(v1 + 24) = 23;
    *(_QWORD *)(v1 + 32) = &unk_503BDA8;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_3503750;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
