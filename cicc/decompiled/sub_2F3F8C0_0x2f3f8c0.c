// Function: sub_2F3F8C0
// Address: 0x2f3f8c0
//
__int64 __fastcall sub_2F3F8C0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 24;
    *(_QWORD *)v1 = "Regalloc eviction policy";
    *(_QWORD *)(v1 + 16) = "regalloc-evict";
    *(_QWORD *)(v1 + 24) = 14;
    *(_QWORD *)(v1 + 32) = &unk_502343C;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_2F3FEB0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
