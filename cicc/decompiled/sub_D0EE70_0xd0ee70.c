// Function: sub_D0EE70
// Address: 0xd0ee70
//
__int64 __fastcall sub_D0EE70(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 22;
    *(_QWORD *)v1 = "CallGraph Construction";
    *(_QWORD *)(v1 + 16) = "basiccg";
    *(_QWORD *)(v1 + 24) = 7;
    *(_QWORD *)(v1 + 32) = &unk_4F86A88;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_D10A20;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
