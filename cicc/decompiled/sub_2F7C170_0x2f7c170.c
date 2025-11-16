// Function: sub_2F7C170
// Address: 0x2f7c170
//
__int64 __fastcall sub_2F7C170(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2F7A2D0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 38;
    *(_QWORD *)v1 = "Register Usage Information Propagation";
    *(_QWORD *)(v1 + 16) = "reg-usage-propagation";
    *(_QWORD *)(v1 + 32) = &unk_5024F4D;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 21;
    *(_QWORD *)(v1 + 48) = sub_2F7C590;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
