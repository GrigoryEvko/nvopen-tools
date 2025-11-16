// Function: sub_CE1440
// Address: 0xce1440
//
__int64 __fastcall sub_CE1440(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_D4AA90();
  sub_B1A2E0((__int64)rwlock);
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 21;
    *(_QWORD *)v1 = "Extra Print Loop Pass";
    *(_QWORD *)(v1 + 16) = "ExtraPrintLoopPass";
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 32) = &unk_4F85150;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_CE1750;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
