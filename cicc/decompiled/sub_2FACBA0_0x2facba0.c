// Function: sub_2FACBA0
// Address: 0x2facba0
//
__int64 __fastcall sub_2FACBA0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 20;
    *(_QWORD *)v1 = "Slot index numbering";
    *(_QWORD *)(v1 + 16) = "slotindexes";
    *(_QWORD *)(v1 + 24) = 11;
    *(_QWORD *)(v1 + 32) = &unk_5025C1C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_2FAD1C0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
