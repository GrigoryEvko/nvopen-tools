// Function: sub_30A3C70
// Address: 0x30a3c70
//
__int64 __fastcall sub_30A3C70(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 14;
    *(_QWORD *)v1 = "DummyCGSCCPass";
    *(_QWORD *)(v1 + 16) = "DummyCGSCCPass";
    *(_QWORD *)(v1 + 24) = 14;
    *(_QWORD *)(v1 + 32) = &unk_502E0CC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_30A4870;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
