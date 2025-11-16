// Function: sub_31CF150
// Address: 0x31cf150
//
__int64 __fastcall sub_31CF150(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 64;
    *(_QWORD *)v1 = "Ensure that the global variables are in the global address space";
    *(_QWORD *)(v1 + 16) = "generic-to-nvvm";
    *(_QWORD *)(v1 + 32) = &unk_5035D70;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 15;
    *(_QWORD *)(v1 + 48) = sub_31CF030;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
