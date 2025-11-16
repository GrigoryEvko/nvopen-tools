// Function: sub_35B5900
// Address: 0x35b5900
//
__int64 __fastcall sub_35B5900(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r13

  sub_2DF86A0((__int64)rwlock);
  sub_2FACF50((__int64)rwlock);
  sub_2E10620((__int64)rwlock);
  sub_2F65EA0((__int64)rwlock);
  sub_2EC54D0((__int64)rwlock);
  sub_2E22F70((__int64)rwlock);
  sub_CF6DB0((__int64)rwlock);
  sub_2E6D3E0((__int64)rwlock);
  sub_2EA61A0((__int64)rwlock);
  sub_300B990((__int64)rwlock);
  sub_2E20C00((__int64)rwlock);
  sub_D84940((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 24;
    *(_QWORD *)v1 = "Basic Register Allocator";
    *(_QWORD *)(v1 + 16) = "regallocbasic";
    *(_QWORD *)(v1 + 32) = &unk_503FDCC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 13;
    *(_QWORD *)(v1 + 48) = sub_35B63F0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
