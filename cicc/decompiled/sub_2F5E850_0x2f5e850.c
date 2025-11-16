// Function: sub_2F5E850
// Address: 0x2f5e850
//
__int64 __fastcall sub_2F5E850(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 24;
    *(_QWORD *)v1 = "Regalloc priority policy";
    *(_QWORD *)(v1 + 16) = "regalloc-priority";
    *(_QWORD *)(v1 + 24) = 17;
    *(_QWORD *)(v1 + 32) = &unk_502442C;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_2F5EDC0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
