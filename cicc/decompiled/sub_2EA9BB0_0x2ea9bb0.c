// Function: sub_2EA9BB0
// Address: 0x2ea9bb0
//
__int64 __fastcall sub_2EA9BB0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 26;
    *(_QWORD *)v1 = "Machine Module Information";
    *(_QWORD *)(v1 + 16) = "machinemoduleinfo";
    *(_QWORD *)(v1 + 24) = 17;
    *(_QWORD *)(v1 + 32) = &unk_50208C0;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_2EAA6F0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
