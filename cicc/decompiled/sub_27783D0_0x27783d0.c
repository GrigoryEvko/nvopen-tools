// Function: sub_27783D0
// Address: 0x27783d0
//
__int64 __fastcall sub_27783D0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_DFEA20((__int64)rwlock);
  sub_CFB980((__int64)rwlock);
  sub_CF6DB0((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_97FFF0((__int64)rwlock);
  sub_103BB40((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 22;
    *(_QWORD *)v1 = "Early CSE w/ MemorySSA";
    *(_QWORD *)(v1 + 16) = "early-cse-memssa";
    *(_QWORD *)(v1 + 32) = &unk_4FFB0EC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 16;
    *(_QWORD *)(v1 + 48) = sub_277BCE0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
