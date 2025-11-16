// Function: sub_2CF93E0
// Address: 0x2cf93e0
//
__int64 __fastcall sub_2CF93E0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 39;
    *(_QWORD *)v1 = "Lower printf to PTX vprintf instruction";
    *(_QWORD *)(v1 + 16) = "nvvm-printf-lowering";
    *(_QWORD *)(v1 + 32) = &unk_50148CC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 20;
    *(_QWORD *)(v1 + 48) = sub_2CF9590;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
