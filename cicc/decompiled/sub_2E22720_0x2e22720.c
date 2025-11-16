// Function: sub_2E22720
// Address: 0x2e22720
//
__int64 __fastcall sub_2E22720(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2FACF50(rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 24;
    *(_QWORD *)v1 = "Live Stack Slot Analysis";
    *(_QWORD *)(v1 + 16) = "livestacks";
    *(_QWORD *)(v1 + 24) = 10;
    *(_QWORD *)(v1 + 32) = &unk_501EB0C;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_2E22FF0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
