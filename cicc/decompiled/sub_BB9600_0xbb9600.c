// Function: sub_BB9600
// Address: 0xbb9600
//
__int64 __fastcall sub_BB9600(__int64 a1, __int64 a2)
{
  pthread_rwlock_t *v2; // rax

  v2 = (pthread_rwlock_t *)sub_BC2B00(a1, a2);
  return sub_BC2DA0(v2);
}
