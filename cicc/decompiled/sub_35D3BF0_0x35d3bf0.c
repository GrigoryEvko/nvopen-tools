// Function: sub_35D3BF0
// Address: 0x35d3bf0
//
__int64 __fastcall sub_35D3BF0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2E44030((__int64)rwlock);
  sub_2E399F0((__int64)rwlock);
  sub_D84940((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 17;
    *(_QWORD *)v1 = "Split static data";
    *(_QWORD *)(v1 + 16) = "static-data-splitter";
    *(_QWORD *)(v1 + 24) = 20;
    *(_QWORD *)(v1 + 32) = &unk_50401EC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_35D40E0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
