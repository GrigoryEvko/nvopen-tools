// Function: sub_35833E0
// Address: 0x35833e0
//
__int64 __fastcall sub_35833E0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 24;
    *(_QWORD *)v1 = "Rename Register Operands";
    *(_QWORD *)(v1 + 16) = "mir-namer";
    *(_QWORD *)(v1 + 32) = &unk_503F24C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 9;
    *(_QWORD *)(v1 + 48) = sub_35832D0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
