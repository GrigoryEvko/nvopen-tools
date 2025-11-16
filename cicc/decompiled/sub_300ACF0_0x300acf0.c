// Function: sub_300ACF0
// Address: 0x300acf0
//
__int64 __fastcall sub_300ACF0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 20;
    *(_QWORD *)v1 = "Virtual Register Map";
    *(_QWORD *)(v1 + 16) = "virtregmap";
    *(_QWORD *)(v1 + 24) = 10;
    *(_QWORD *)(v1 + 32) = &unk_502A66C;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_300AEC0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
