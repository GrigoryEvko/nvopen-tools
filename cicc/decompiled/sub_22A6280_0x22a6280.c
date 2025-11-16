// Function: sub_22A6280
// Address: 0x22a6280
//
__int64 __fastcall sub_22A6280(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 27;
    *(_QWORD *)v1 = "DXIL Resource Type Analysis";
    *(_QWORD *)(v1 + 16) = "dxil-resource-type";
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 32) = &unk_4FDB68C;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_22A9A30;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
