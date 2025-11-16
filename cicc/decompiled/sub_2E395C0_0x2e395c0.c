// Function: sub_2E395C0
// Address: 0x2e395c0
//
__int64 __fastcall sub_2E395C0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2E44030(rwlock);
  sub_2EA61A0(rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 32;
    *(_QWORD *)v1 = "Machine Block Frequency Analysis";
    *(_QWORD *)(v1 + 16) = "machine-block-freq";
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 32) = &unk_501EC08;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_2E39B90;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
