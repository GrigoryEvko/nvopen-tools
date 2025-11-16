// Function: sub_22E4970
// Address: 0x22e4970
//
__int64 __fastcall sub_22E4970(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 24;
    *(_QWORD *)v1 = "View regions of function";
    *(_QWORD *)(v1 + 16) = "view-regions";
    *(_QWORD *)(v1 + 32) = &unk_4FDC061;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 24) = 12;
    *(_QWORD *)(v1 + 48) = sub_22E56B0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
