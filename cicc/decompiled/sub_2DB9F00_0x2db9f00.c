// Function: sub_2DB9F00
// Address: 0x2db9f00
//
__int64 __fastcall sub_2DB9F00(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 20;
    *(_QWORD *)v1 = "Expand large div/rem";
    *(_QWORD *)(v1 + 16) = "expand-large-div-rem";
    *(_QWORD *)(v1 + 32) = &unk_501D20C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 20;
    *(_QWORD *)(v1 + 48) = sub_2DBA230;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
