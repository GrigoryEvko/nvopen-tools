// Function: sub_F41E00
// Address: 0xf41e00
//
__int64 __fastcall sub_F41E00(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 27;
    *(_QWORD *)v1 = "Break critical edges in CFG";
    *(_QWORD *)(v1 + 16) = "break-crit-edges";
    *(_QWORD *)(v1 + 32) = &unk_4F8BE8C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 16;
    *(_QWORD *)(v1 + 48) = sub_F42420;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
