// Function: sub_2D229E0
// Address: 0x2d229e0
//
__int64 __fastcall sub_2D229E0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 28;
    *(_QWORD *)v1 = "Assignment Tracking Analysis";
    *(_QWORD *)(v1 + 16) = "debug-ata";
    *(_QWORD *)(v1 + 24) = 9;
    *(_QWORD *)(v1 + 32) = &unk_50165D0;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_2D286E0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
