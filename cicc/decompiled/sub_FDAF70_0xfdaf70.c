// Function: sub_FDAF70
// Address: 0xfdaf70
//
__int64 __fastcall sub_FDAF70(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_FEEC20();
  sub_D4AA90((__int64)rwlock);
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 24;
    *(_QWORD *)v1 = "Block Frequency Analysis";
    *(_QWORD *)(v1 + 16) = "block-freq";
    *(_QWORD *)(v1 + 24) = 10;
    *(_QWORD *)(v1 + 32) = &unk_4F8D9B0;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_FDC710;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
