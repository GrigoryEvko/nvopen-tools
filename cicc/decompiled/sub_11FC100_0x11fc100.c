// Function: sub_11FC100
// Address: 0x11fc100
//
__int64 __fastcall sub_11FC100(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 19;
    *(_QWORD *)v1 = "Cycle Info Analysis";
    *(_QWORD *)(v1 + 16) = "cycles";
    *(_QWORD *)(v1 + 24) = 6;
    *(_QWORD *)(v1 + 32) = &unk_4F92384;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_11FC7E0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
