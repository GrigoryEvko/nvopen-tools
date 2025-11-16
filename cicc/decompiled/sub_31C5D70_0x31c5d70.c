// Function: sub_31C5D70
// Address: 0x31c5d70
//
__int64 __fastcall sub_31C5D70(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 35;
    *(_QWORD *)v1 = "Handle 'n' asm constraint for NVPTX";
    *(_QWORD *)(v1 + 16) = "asm-constraint";
    *(_QWORD *)(v1 + 32) = &unk_5035D5C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 14;
    *(_QWORD *)(v1 + 48) = sub_31C5DF0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
