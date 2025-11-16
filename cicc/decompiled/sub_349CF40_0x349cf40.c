// Function: sub_349CF40
// Address: 0x349cf40
//
__int64 __fastcall sub_349CF40(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 25;
    *(_QWORD *)v1 = "Live DEBUG_VALUE analysis";
    *(_QWORD *)(v1 + 16) = "livedebugvalues";
    *(_QWORD *)(v1 + 32) = &unk_503A0EC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 15;
    *(_QWORD *)(v1 + 48) = sub_349D300;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
