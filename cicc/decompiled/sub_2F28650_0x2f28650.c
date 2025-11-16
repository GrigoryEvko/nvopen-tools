// Function: sub_2F28650
// Address: 0x2f28650
//
__int64 __fastcall sub_2F28650(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 44;
    *(_QWORD *)v1 = "Implement the 'patchable-function' attribute";
    *(_QWORD *)(v1 + 16) = "patchable-function";
    *(_QWORD *)(v1 + 32) = &unk_50226EC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 48) = sub_2F28B80;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
