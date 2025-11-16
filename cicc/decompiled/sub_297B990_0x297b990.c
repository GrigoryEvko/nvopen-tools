// Function: sub_297B990
// Address: 0x297b990
//
__int64 __fastcall sub_297B990(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  sub_D9A960((__int64)rwlock);
  sub_DFEA20((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 32;
    *(_QWORD *)v1 = "Straight line strength reduction";
    *(_QWORD *)(v1 + 16) = "slsr";
    *(_QWORD *)(v1 + 32) = &unk_500732C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 4;
    *(_QWORD *)(v1 + 48) = sub_297E650;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
