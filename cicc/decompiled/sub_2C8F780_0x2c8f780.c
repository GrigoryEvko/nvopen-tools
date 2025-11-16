// Function: sub_2C8F780
// Address: 0x2c8f780
//
__int64 __fastcall sub_2C8F780(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 43;
    *(_QWORD *)v1 = "BypassSlowDivision to bypass slow divisions";
    *(_QWORD *)(v1 + 16) = "bypass-slow-division";
    *(_QWORD *)(v1 + 32) = &unk_5011B4C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 20;
    *(_QWORD *)(v1 + 48) = sub_2C8F690;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
