// Function: sub_2FEE120
// Address: 0x2fee120
//
__int64 __fastcall sub_2FEE120(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 25;
    *(_QWORD *)v1 = "Target Pass Configuration";
    *(_QWORD *)(v1 + 16) = "targetpassconfig";
    *(_QWORD *)(v1 + 24) = 16;
    *(_QWORD *)(v1 + 32) = &unk_5027190;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_2FF06F0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
