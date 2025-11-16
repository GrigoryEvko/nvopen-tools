// Function: sub_37EB970
// Address: 0x37eb970
//
__int64 __fastcall sub_37EB970(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 52;
    *(_QWORD *)v1 = "Check CFA info and insert CFI instructions if needed";
    *(_QWORD *)(v1 + 16) = "cfi-instr-inserter";
    *(_QWORD *)(v1 + 32) = &unk_5051344;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 48) = sub_37EE9A0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
