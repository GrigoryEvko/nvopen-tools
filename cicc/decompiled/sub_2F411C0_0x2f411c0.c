// Function: sub_2F411C0
// Address: 0x2f411c0
//
__int64 __fastcall sub_2F411C0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 23;
    *(_QWORD *)v1 = "Fast Register Allocator";
    *(_QWORD *)(v1 + 16) = "regallocfast";
    *(_QWORD *)(v1 + 32) = &unk_502387C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 12;
    *(_QWORD *)(v1 + 48) = sub_2F42D20;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
