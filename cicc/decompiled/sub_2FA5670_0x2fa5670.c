// Function: sub_2FA5670
// Address: 0x2fa5670
//
__int64 __fastcall sub_2FA5670(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2DD02F0((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 24;
    *(_QWORD *)v1 = "Shadow Stack GC Lowering";
    *(_QWORD *)(v1 + 16) = "shadow-stack-gc-lowering";
    *(_QWORD *)(v1 + 32) = &unk_5025C0C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 24;
    *(_QWORD *)(v1 + 48) = sub_2FA5C00;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
