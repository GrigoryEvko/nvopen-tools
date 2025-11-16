// Function: sub_34C9990
// Address: 0x34c9990
//
__int64 __fastcall sub_34C9990(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 46;
    *(_QWORD *)v1 = "Insert CFI remember/restore state instructions";
    *(_QWORD *)(v1 + 16) = "cfi-fixup";
    *(_QWORD *)(v1 + 24) = 9;
    *(_QWORD *)(v1 + 32) = &unk_503AD4C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_34C9D10;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
