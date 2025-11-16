// Function: sub_BB95C0
// Address: 0xbb95c0
//
__int64 __fastcall sub_BB95C0(__int64 a1, __int64 a2)
{
  pthread_rwlock_t *v2; // rax
  pthread_rwlock_t *v4; // rdi
  __int64 v5; // rax

  v2 = (pthread_rwlock_t *)sub_BC2B00(a1, a2);
  v4 = v2;
  v5 = sub_BC2C30(v2);
  if ( v5 )
    return (*(__int64 (__fastcall **)(pthread_rwlock_t *, __int64))(v5 + 48))(v4, a1);
  else
    return 0;
}
