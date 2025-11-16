// Function: sub_35320A0
// Address: 0x35320a0
//
__int64 __fastcall sub_35320A0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 25;
    *(_QWORD *)v1 = "Machine Function Outliner";
    *(_QWORD *)(v1 + 16) = "machine-outliner";
    *(_QWORD *)(v1 + 32) = &unk_503D78C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 16;
    *(_QWORD *)(v1 + 48) = sub_3534A50;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
