// Function: sub_2EAFAB0
// Address: 0x2eafab0
//
__int64 __fastcall sub_2EAFAB0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_35035A0();
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 35;
    *(_QWORD *)v1 = "Machine Optimization Remark Emitter";
    *(_QWORD *)(v1 + 16) = "machine-opt-remark-emitter";
    *(_QWORD *)(v1 + 24) = 26;
    *(_QWORD *)(v1 + 32) = &unk_50209AC;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_2EAFE50;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
