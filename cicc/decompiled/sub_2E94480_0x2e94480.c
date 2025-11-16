// Function: sub_2E94480
// Address: 0x2e94480
//
__int64 __fastcall sub_2E94480(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 38;
    *(_QWORD *)v1 = "Machine Late Instructions Cleanup Pass";
    *(_QWORD *)(v1 + 16) = "machine-latecleanup";
    *(_QWORD *)(v1 + 32) = &unk_50201DC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 19;
    *(_QWORD *)(v1 + 48) = sub_2E94A00;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
