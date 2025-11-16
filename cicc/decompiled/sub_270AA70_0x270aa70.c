// Function: sub_270AA70
// Address: 0x270aa70
//
__int64 __fastcall sub_270AA70(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_CF6DB0((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 20;
    *(_QWORD *)v1 = "ObjC ARC contraction";
    *(_QWORD *)(v1 + 16) = "objc-arc-contract";
    *(_QWORD *)(v1 + 32) = &unk_4FF9A24;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 17;
    *(_QWORD *)(v1 + 48) = sub_270B3B0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
