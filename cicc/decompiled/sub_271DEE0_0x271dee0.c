// Function: sub_271DEE0
// Address: 0x271dee0
//
__int64 __fastcall sub_271DEE0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_104C240((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 32;
    *(_QWORD *)v1 = "Aggressive Dead Code Elimination";
    *(_QWORD *)(v1 + 16) = "adce";
    *(_QWORD *)(v1 + 32) = &unk_4FF9B0C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 4;
    *(_QWORD *)(v1 + 48) = sub_271EBE0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
