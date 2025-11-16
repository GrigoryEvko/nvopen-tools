// Function: sub_36D0C60
// Address: 0x36d0c60
//
__int64 __fastcall sub_36D0C60(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 20;
    *(_QWORD *)v1 = "NVPTX Forward Params";
    *(_QWORD *)(v1 + 16) = "nvptx-forward-params";
    *(_QWORD *)(v1 + 32) = &unk_5040BDC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 20;
    *(_QWORD *)(v1 + 48) = sub_36D1350;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
