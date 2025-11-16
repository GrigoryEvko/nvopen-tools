// Function: sub_2DD1B10
// Address: 0x2dd1b10
//
__int64 __fastcall sub_2DD1B10(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2DD02F0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 11;
    *(_QWORD *)v1 = "GC Lowering";
    *(_QWORD *)(v1 + 16) = "gc-lowering";
    *(_QWORD *)(v1 + 32) = &unk_501DA25;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 11;
    *(_QWORD *)(v1 + 48) = sub_2DD2700;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
