// Function: sub_22E4850
// Address: 0x22e4850
//
__int64 __fastcall sub_22E4850(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 39;
    *(_QWORD *)v1 = "Print regions of function to 'dot' file";
    *(_QWORD *)(v1 + 16) = "dot-regions";
    *(_QWORD *)(v1 + 32) = &unk_4FDC063;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 24) = 11;
    *(_QWORD *)(v1 + 48) = sub_22E5310;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
