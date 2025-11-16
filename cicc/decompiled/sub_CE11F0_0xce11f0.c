// Function: sub_CE11F0
// Address: 0xce11f0
//
__int64 __fastcall sub_CE11F0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_FCE0C0();
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 25;
    *(_QWORD *)v1 = "Extra Print Function Pass";
    *(_QWORD *)(v1 + 16) = "ExtraPrintFunctionPass";
    *(_QWORD *)(v1 + 24) = 22;
    *(_QWORD *)(v1 + 32) = &unk_4F85158;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_CE1280;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
