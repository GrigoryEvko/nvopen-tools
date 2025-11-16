// Function: sub_2F7E500
// Address: 0x2f7e500
//
__int64 __fastcall sub_2F7E500(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2FACF50(rwlock);
  sub_2E10620((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 31;
    *(_QWORD *)v1 = "Rename Independent Subregisters";
    *(_QWORD *)(v1 + 16) = "rename-independent-subregs";
    *(_QWORD *)(v1 + 32) = &unk_5024F60;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 26;
    *(_QWORD *)(v1 + 48) = sub_2F7E3F0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
