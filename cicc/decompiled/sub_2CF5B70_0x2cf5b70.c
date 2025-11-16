// Function: sub_2CF5B70
// Address: 0x2cf5b70
//
__int64 __fastcall sub_2CF5B70(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 20;
    *(_QWORD *)v1 = "Normalize 64-bit Gep";
    *(_QWORD *)(v1 + 16) = "Normalize-Gep";
    *(_QWORD *)(v1 + 32) = &unk_5014629;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 13;
    *(_QWORD *)(v1 + 48) = sub_2CF6300;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
