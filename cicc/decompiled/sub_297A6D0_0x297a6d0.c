// Function: sub_297A6D0
// Address: 0x297a6d0
//
__int64 __fastcall sub_297A6D0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_DFEA20((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 34;
    *(_QWORD *)v1 = "Speculatively execute instructions";
    *(_QWORD *)(v1 + 16) = "speculative-execution";
    *(_QWORD *)(v1 + 32) = &unk_500708C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 21;
    *(_QWORD *)(v1 + 48) = sub_297B440;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
