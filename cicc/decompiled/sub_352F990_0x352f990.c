// Function: sub_352F990
// Address: 0x352f990
//
__int64 __fastcall sub_352F990(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 49;
    *(_QWORD *)v1 = "Split machine functions using profile information";
    *(_QWORD *)(v1 + 16) = "machine-function-splitter";
    *(_QWORD *)(v1 + 32) = &unk_503D4F4;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 25;
    *(_QWORD *)(v1 + 48) = sub_352FEE0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
