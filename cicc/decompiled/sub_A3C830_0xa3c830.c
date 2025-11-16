// Function: sub_A3C830
// Address: 0xa3c830
//
__int64 __fastcall sub_A3C830(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_D78240();
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 13;
    *(_QWORD *)v1 = "Write Bitcode";
    *(_QWORD *)(v1 + 16) = "write-bitcode";
    *(_QWORD *)(v1 + 32) = &unk_4F8090C;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 24) = 13;
    *(_QWORD *)(v1 + 48) = sub_A3CCC0;
  }
  sub_BC3090(rwlock);
  return v2;
}
