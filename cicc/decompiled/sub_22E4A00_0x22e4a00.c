// Function: sub_22E4A00
// Address: 0x22e4a00
//
__int64 __fastcall sub_22E4A00(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 50;
    *(_QWORD *)v1 = "View regions of function (with no function bodies)";
    *(_QWORD *)(v1 + 16) = "view-regions-only";
    *(_QWORD *)(v1 + 32) = &unk_4FDC060;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 24) = 17;
    *(_QWORD *)(v1 + 48) = sub_22E5880;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
