// Function: sub_300BA10
// Address: 0x300ba10
//
__int64 __fastcall sub_300BA10(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2FACF50((__int64)rwlock);
  sub_2E10620((__int64)rwlock);
  sub_2DF86A0((__int64)rwlock);
  sub_2E20C00((__int64)rwlock);
  sub_2E22F70((__int64)rwlock);
  sub_300B990((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 25;
    *(_QWORD *)v1 = "Virtual Register Rewriter";
    *(_QWORD *)(v1 + 16) = "virtregrewriter";
    *(_QWORD *)(v1 + 32) = &unk_502A65C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 15;
    *(_QWORD *)(v1 + 48) = sub_300B090;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
