// Function: sub_CFB060
// Address: 0xcfb060
//
__int64 __fastcall sub_CFB060(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 24;
    *(_QWORD *)v1 = "Assumption Cache Tracker";
    *(_QWORD *)(v1 + 16) = "assumption-cache-tracker";
    *(_QWORD *)(v1 + 24) = 24;
    *(_QWORD *)(v1 + 32) = &unk_4F8662C;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_CFBB10;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
