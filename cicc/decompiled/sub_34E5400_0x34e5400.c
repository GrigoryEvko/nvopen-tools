// Function: sub_34E5400
// Address: 0x34e5400
//
__int64 __fastcall sub_34E5400(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 29;
    *(_QWORD *)v1 = "Contiguously Lay Out Funclets";
    *(_QWORD *)(v1 + 16) = "funclet-layout";
    *(_QWORD *)(v1 + 32) = &unk_503B12C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 14;
    *(_QWORD *)(v1 + 48) = sub_34E5A10;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
