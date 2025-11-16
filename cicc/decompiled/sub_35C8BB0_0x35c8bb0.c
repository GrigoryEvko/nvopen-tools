// Function: sub_35C8BB0
// Address: 0x35c8bb0
//
__int64 __fastcall sub_35C8BB0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 33;
    *(_QWORD *)v1 = "Machine Sanitizer Binary Metadata";
    *(_QWORD *)(v1 + 16) = "machine-sanmd";
    *(_QWORD *)(v1 + 32) = &unk_503FF3E;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 13;
    *(_QWORD *)(v1 + 48) = sub_35C9100;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
