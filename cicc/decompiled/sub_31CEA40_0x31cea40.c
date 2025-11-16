// Function: sub_31CEA40
// Address: 0x31cea40
//
__int64 __fastcall sub_31CEA40(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 41;
    *(_QWORD *)v1 = "Libdevice library specific checking phase";
    *(_QWORD *)(v1 + 16) = "nvvm-libdevice-check";
    *(_QWORD *)(v1 + 24) = 20;
    *(_QWORD *)(v1 + 32) = &unk_4CE0098;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_31CEB20;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
