// Function: sub_2F4C2E0
// Address: 0x2f4c2e0
//
__int64 __fastcall sub_2F4C2E0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r13

  sub_2DF86A0((__int64)rwlock);
  sub_2FACF50(rwlock);
  sub_2E10620((__int64)rwlock);
  sub_2F65EA0(rwlock);
  sub_2EC54D0((__int64)rwlock);
  sub_2E22F70((__int64)rwlock);
  sub_2E6D3E0((__int64)rwlock);
  sub_2EA61A0((__int64)rwlock);
  sub_300B990(rwlock);
  sub_2E20C00((__int64)rwlock);
  sub_2DB8D80((__int64)rwlock);
  sub_2FAECB0(rwlock);
  sub_2EAFCC0((__int64)rwlock);
  sub_2F403D0((__int64)rwlock);
  sub_2F5F4E0(rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 25;
    *(_QWORD *)v1 = "Greedy Register Allocator";
    *(_QWORD *)(v1 + 16) = "greedy";
    *(_QWORD *)(v1 + 32) = &unk_5023990;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 6;
    *(_QWORD *)(v1 + 48) = sub_2F50470;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
