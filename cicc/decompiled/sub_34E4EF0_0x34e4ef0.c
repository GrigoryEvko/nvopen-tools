// Function: sub_34E4EF0
// Address: 0x34e4ef0
//
__int64 __fastcall sub_34E4EF0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 19;
    *(_QWORD *)v1 = "Insert fentry calls";
    *(_QWORD *)(v1 + 16) = "fentry-insert";
    *(_QWORD *)(v1 + 32) = &unk_503B124;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 13;
    *(_QWORD *)(v1 + 48) = sub_34E52C0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
