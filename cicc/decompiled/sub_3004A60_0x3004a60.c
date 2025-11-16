// Function: sub_3004A60
// Address: 0x3004a60
//
__int64 __fastcall sub_3004A60(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 39;
    *(_QWORD *)v1 = "Remove unreachable machine basic blocks";
    *(_QWORD *)(v1 + 16) = "unreachable-mbb-elimination";
    *(_QWORD *)(v1 + 32) = &unk_502A64C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 27;
    *(_QWORD *)(v1 + 48) = sub_30048D0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
