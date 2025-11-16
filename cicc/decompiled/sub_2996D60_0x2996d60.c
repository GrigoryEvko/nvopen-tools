// Function: sub_2996D60
// Address: 0x2996d60
//
__int64 __fastcall sub_2996D60(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_DFEA20((__int64)rwlock);
  sub_1049990((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 21;
    *(_QWORD *)v1 = "Tail Call Elimination";
    *(_QWORD *)(v1 + 16) = "tailcallelim";
    *(_QWORD *)(v1 + 32) = &unk_500794C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 12;
    *(_QWORD *)(v1 + 48) = sub_2998210;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
