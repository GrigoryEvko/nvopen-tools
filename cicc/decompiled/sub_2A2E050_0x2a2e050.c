// Function: sub_2A2E050
// Address: 0x2a2e050
//
__int64 __fastcall sub_2A2E050(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 43;
    *(_QWORD *)v1 = "Lower @llvm.global_dtors via `__cxa_atexit`";
    *(_QWORD *)(v1 + 16) = "lower-global-dtors";
    *(_QWORD *)(v1 + 32) = &unk_500A96C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 48) = sub_2A2E2F0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
