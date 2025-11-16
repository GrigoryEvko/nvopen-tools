// Function: sub_22E48E0
// Address: 0x22e48e0
//
__int64 __fastcall sub_22E48E0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 65;
    *(_QWORD *)v1 = "Print regions of function to 'dot' file (with no function bodies)";
    *(_QWORD *)(v1 + 16) = "dot-regions-only";
    *(_QWORD *)(v1 + 32) = &unk_4FDC062;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 24) = 16;
    *(_QWORD *)(v1 + 48) = sub_22E54E0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
