// Function: sub_34BBF90
// Address: 0x34bbf90
//
__int64 __fastcall sub_34BBF90(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2D50D40((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 88;
    *(_QWORD *)v1 = "Prepares for basic block sections, by splitting functions into clusters of basic blocks.";
    *(_QWORD *)(v1 + 16) = "bbsections-prepare";
    *(_QWORD *)(v1 + 32) = &unk_503A634;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 48) = sub_34BC510;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
