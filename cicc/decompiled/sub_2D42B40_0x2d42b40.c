// Function: sub_2D42B40
// Address: 0x2d42b40
//
__int64 __fastcall sub_2D42B40(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2FEF6D0();
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 26;
    *(_QWORD *)v1 = "Expand Atomic instructions";
    *(_QWORD *)(v1 + 16) = "atomic-expand";
    *(_QWORD *)(v1 + 32) = &unk_501694C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 13;
    *(_QWORD *)(v1 + 48) = sub_2D45E90;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
