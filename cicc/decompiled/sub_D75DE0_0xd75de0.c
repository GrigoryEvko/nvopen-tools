// Function: sub_D75DE0
// Address: 0xd75de0
//
__int64 __fastcall sub_D75DE0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 19;
    *(_QWORD *)v1 = "Module summary info";
    *(_QWORD *)(v1 + 16) = "module-summary-info";
    *(_QWORD *)(v1 + 24) = 19;
    *(_QWORD *)(v1 + 32) = &unk_4F8780C;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_D78550;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
