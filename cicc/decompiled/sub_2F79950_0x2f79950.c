// Function: sub_2F79950
// Address: 0x2f79950
//
__int64 __fastcall sub_2F79950(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 34;
    *(_QWORD *)v1 = "Register Usage Information Storage";
    *(_QWORD *)(v1 + 16) = "reg-usage-info";
    *(_QWORD *)(v1 + 24) = 14;
    *(_QWORD *)(v1 + 32) = &unk_5024E70;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_2F7A350;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
