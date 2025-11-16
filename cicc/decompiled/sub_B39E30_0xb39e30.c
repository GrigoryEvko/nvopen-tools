// Function: sub_B39E30
// Address: 0xb39e30
//
__int64 __fastcall sub_B39E30(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 24;
    *(_QWORD *)v1 = "Print function to stderr";
    *(_QWORD *)(v1 + 16) = "print-function";
    *(_QWORD *)(v1 + 32) = &unk_4F816EC;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 24) = 14;
    *(_QWORD *)(v1 + 48) = sub_B3A120;
  }
  sub_BC3090(rwlock);
  return v2;
}
