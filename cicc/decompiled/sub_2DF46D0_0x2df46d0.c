// Function: sub_2DF46D0
// Address: 0x2df46d0
//
__int64 __fastcall sub_2DF46D0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2E6D3E0(rwlock);
  sub_2E10620(rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 23;
    *(_QWORD *)v1 = "Debug Variable Analysis";
    *(_QWORD *)(v1 + 16) = "livedebugvars";
    *(_QWORD *)(v1 + 24) = 13;
    *(_QWORD *)(v1 + 32) = &unk_501E91C;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_2DF8830;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
