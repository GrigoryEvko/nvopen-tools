// Function: sub_22A6310
// Address: 0x22a6310
//
__int64 __fastcall sub_22A6310(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 30;
    *(_QWORD *)v1 = "DXIL Resource Binding Analysis";
    *(_QWORD *)(v1 + 16) = "dxil-resource-binding";
    *(_QWORD *)(v1 + 24) = 21;
    *(_QWORD *)(v1 + 32) = &unk_4FDB685;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_22A9C00;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
