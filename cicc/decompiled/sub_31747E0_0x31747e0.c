// Function: sub_31747E0
// Address: 0x31747e0
//
__int64 __fastcall sub_31747E0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 20;
    *(_QWORD *)v1 = "A No-Op Barrier Pass";
    *(_QWORD *)(v1 + 16) = "barrier";
    *(_QWORD *)(v1 + 32) = &unk_503440C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 7;
    *(_QWORD *)(v1 + 48) = sub_31748E0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
