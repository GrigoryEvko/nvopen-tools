// Function: sub_2F9A560
// Address: 0x2f9a560
//
__int64 __fastcall sub_2F9A560(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_D4AA90((__int64)rwlock);
  sub_D84940((__int64)rwlock);
  sub_2FEF6D0(rwlock);
  sub_DFEA20((__int64)rwlock);
  sub_FDC5A0((__int64)rwlock);
  sub_1049990((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 16;
    *(_QWORD *)v1 = "Optimize selects";
    *(_QWORD *)(v1 + 16) = "select-optimize";
    *(_QWORD *)(v1 + 32) = &unk_50255EC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 15;
    *(_QWORD *)(v1 + 48) = sub_2F9C060;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
