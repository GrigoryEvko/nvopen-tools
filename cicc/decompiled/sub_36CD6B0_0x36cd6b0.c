// Function: sub_36CD6B0
// Address: 0x36cd6b0
//
__int64 __fastcall sub_36CD6B0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 51;
    *(_QWORD *)v1 = "Lower atomics of local memory to simple load/stores";
    *(_QWORD *)(v1 + 16) = "nvptx-atomic-lower";
    *(_QWORD *)(v1 + 32) = &unk_504090C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 48) = sub_36CD7A0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
