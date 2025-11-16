// Function: sub_31CD280
// Address: 0x31cd280
//
__int64 __fastcall sub_31CD280(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 27;
    *(_QWORD *)v1 = "Propagate access properties";
    *(_QWORD *)(v1 + 16) = "prop-ap";
    *(_QWORD *)(v1 + 24) = 7;
    *(_QWORD *)(v1 + 32) = &unk_5035D64;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_31CE080;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
