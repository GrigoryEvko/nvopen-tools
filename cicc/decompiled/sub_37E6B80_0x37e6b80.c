// Function: sub_37E6B80
// Address: 0x37e6b80
//
__int64 __fastcall sub_37E6B80(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 22;
    *(_QWORD *)v1 = "Branch relaxation pass";
    *(_QWORD *)(v1 + 16) = "branch-relaxation";
    *(_QWORD *)(v1 + 32) = &unk_505132C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 17;
    *(_QWORD *)(v1 + 48) = sub_37E7280;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
