// Function: sub_22BD970
// Address: 0x22bd970
//
__int64 __fastcall sub_22BD970(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_CFB980((__int64)rwlock);
  sub_97FFF0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 31;
    *(_QWORD *)v1 = "Lazy Value Information Analysis";
    *(_QWORD *)(v1 + 16) = "lazy-value-info";
    *(_QWORD *)(v1 + 24) = 15;
    *(_QWORD *)(v1 + 32) = &unk_4FDBCD4;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_22C11D0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
