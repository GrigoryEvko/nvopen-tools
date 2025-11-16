// Function: sub_29D33E0
// Address: 0x29d33e0
//
__int64 __fastcall sub_29D33E0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  sub_D4AA90((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 51;
    *(_QWORD *)v1 = "Convert irreducible control-flow into natural loops";
    *(_QWORD *)(v1 + 16) = "fix-irreducible";
    *(_QWORD *)(v1 + 32) = &unk_5009014;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 15;
    *(_QWORD *)(v1 + 48) = sub_29D3AA0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
