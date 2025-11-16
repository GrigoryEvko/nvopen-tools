// Function: sub_22D45E0
// Address: 0x22d45e0
//
__int64 __fastcall sub_22D45E0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 19;
    *(_QWORD *)v1 = "Phi Values Analysis";
    *(_QWORD *)(v1 + 16) = "phi-values";
    *(_QWORD *)(v1 + 24) = 10;
    *(_QWORD *)(v1 + 32) = &unk_4FDBCF4;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_22D5E30;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
