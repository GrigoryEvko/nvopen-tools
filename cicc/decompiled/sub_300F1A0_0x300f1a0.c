// Function: sub_300F1A0
// Address: 0x300f1a0
//
__int64 __fastcall sub_300F1A0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 30;
    *(_QWORD *)v1 = "Prepare WebAssembly exceptions";
    *(_QWORD *)(v1 + 16) = "wasm-eh-prepare";
    *(_QWORD *)(v1 + 32) = &unk_502A674;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 15;
    *(_QWORD *)(v1 + 48) = sub_300F280;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
