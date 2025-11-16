// Function: sub_3580F70
// Address: 0x3580f70
//
__int64 __fastcall sub_3580F70(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 37;
    *(_QWORD *)v1 = "Add MIR Flow Sensitive Discriminators";
    *(_QWORD *)(v1 + 16) = "mirfs-discriminators";
    *(_QWORD *)(v1 + 24) = 20;
    *(_QWORD *)(v1 + 32) = &unk_503F16C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_3581050;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
