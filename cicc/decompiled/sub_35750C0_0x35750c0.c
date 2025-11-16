// Function: sub_35750C0
// Address: 0x35750c0
//
__int64 __fastcall sub_35750C0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_3574F00((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 38;
    *(_QWORD *)v1 = "Print Machine Uniformity Info Analysis";
    *(_QWORD *)(v1 + 16) = "print-machine-uniformity";
    *(_QWORD *)(v1 + 32) = &unk_503F084;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 24) = 24;
    *(_QWORD *)(v1 + 48) = sub_35751D0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
