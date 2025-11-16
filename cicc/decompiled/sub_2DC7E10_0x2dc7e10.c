// Function: sub_2DC7E10
// Address: 0x2dc7e10
//
__int64 __fastcall sub_2DC7E10(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 41;
    *(_QWORD *)v1 = "Post-RA pseudo instruction expansion pass";
    *(_QWORD *)(v1 + 16) = "postrapseudos";
    *(_QWORD *)(v1 + 32) = &unk_501D66C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 13;
    *(_QWORD *)(v1 + 48) = sub_2DC8AA0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
