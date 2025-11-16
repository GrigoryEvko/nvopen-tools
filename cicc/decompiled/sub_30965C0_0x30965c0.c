// Function: sub_30965C0
// Address: 0x30965c0
//
__int64 __fastcall sub_30965C0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 41;
    *(_QWORD *)v1 = "Optimize redundant ANDb16ri instrunctions";
    *(_QWORD *)(v1 + 16) = "nvptx-trunc-opts";
    *(_QWORD *)(v1 + 32) = &unk_502D52C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 16;
    *(_QWORD *)(v1 + 48) = sub_3096830;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
