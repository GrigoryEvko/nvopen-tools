// Function: sub_34E5B60
// Address: 0x34e5b60
//
__int64 __fastcall sub_34E5B60(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 80;
    *(_QWORD *)v1 = "Removes empty basic blocks and redirects their uses to their fallthrough blocks.";
    *(_QWORD *)(v1 + 16) = "gc-empty-basic-blocks";
    *(_QWORD *)(v1 + 24) = 21;
    *(_QWORD *)(v1 + 32) = &unk_503B134;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_34E5F50;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
