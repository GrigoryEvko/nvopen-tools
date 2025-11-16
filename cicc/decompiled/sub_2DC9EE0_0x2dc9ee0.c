// Function: sub_2DC9EE0
// Address: 0x2dc9ee0
//
__int64 __fastcall sub_2DC9EE0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 44;
    *(_QWORD *)v1 = "Finalize ISel and expand pseudo-instructions";
    *(_QWORD *)(v1 + 16) = "finalize-isel";
    *(_QWORD *)(v1 + 32) = &unk_501D67C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 13;
    *(_QWORD *)(v1 + 48) = sub_2DC9DD0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
