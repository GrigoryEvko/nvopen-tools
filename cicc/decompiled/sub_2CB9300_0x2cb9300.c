// Function: sub_2CB9300
// Address: 0x2cb9300
//
__int64 __fastcall sub_2CB9300(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 35;
    *(_QWORD *)v1 = "Mark must inline functions for NVVM";
    *(_QWORD *)(v1 + 16) = "nv-inline-must";
    *(_QWORD *)(v1 + 32) = &unk_50133CC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 14;
    *(_QWORD *)(v1 + 48) = sub_2CB9F60;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
