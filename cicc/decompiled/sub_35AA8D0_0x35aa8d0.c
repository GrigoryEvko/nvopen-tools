// Function: sub_35AA8D0
// Address: 0x35aa8d0
//
__int64 __fastcall sub_35AA8D0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 28;
    *(_QWORD *)v1 = "Process Implicit Definitions";
    *(_QWORD *)(v1 + 16) = "processimpdefs";
    *(_QWORD *)(v1 + 32) = &unk_503FCF4;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 14;
    *(_QWORD *)(v1 + 48) = sub_35AAC30;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
