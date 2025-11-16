// Function: sub_2D02600
// Address: 0x2d02600
//
__int64 __fastcall sub_2D02600(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 53;
    *(_QWORD *)v1 = "Replace occurences of __nvvm_reflect() calls with 0/1";
    *(_QWORD *)(v1 + 16) = "nvvm-reflect";
    *(_QWORD *)(v1 + 32) = &unk_5014D64;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 12;
    *(_QWORD *)(v1 + 48) = sub_2D02B50;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
