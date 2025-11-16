// Function: sub_2D19950
// Address: 0x2d19950
//
__int64 __fastcall sub_2D19950(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 53;
    *(_QWORD *)v1 = "Find good alignment for statically sized local arrays";
    *(_QWORD *)(v1 + 16) = "set-local-array-alignment";
    *(_QWORD *)(v1 + 32) = &unk_5016124;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 25;
    *(_QWORD *)(v1 + 48) = sub_2D1A2A0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
