// Function: sub_BB8680
// Address: 0xbb8680
//
const char *__fastcall sub_BB8680(__int64 a1, __int64 a2)
{
  pthread_rwlock_t *v2; // rax
  __int64 v3; // rax
  const char *v4; // r8

  v2 = (pthread_rwlock_t *)sub_BC2B00(a1, a2);
  v3 = sub_BC2C30(v2);
  v4 = "Unnamed pass: implement Pass::getPassName()";
  if ( v3 )
    return *(const char **)v3;
  return v4;
}
