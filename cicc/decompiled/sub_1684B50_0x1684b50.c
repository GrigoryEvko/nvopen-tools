// Function: sub_1684B50
// Address: 0x1684b50
//
int __fastcall sub_1684B50(pthread_mutex_t **a1)
{
  pthread_mutex_t *v2; // rdi
  __int64 v4; // rdi

  v2 = *a1;
  if ( v2 )
    return j__pthread_mutex_lock(v2);
  sub_1688E30();
  if ( !*a1 )
  {
    v4 = sub_1683C60(0);
    *a1 = (pthread_mutex_t *)sub_16890B0();
    sub_1683C60(v4);
  }
  sub_1688E70();
  return j__pthread_mutex_lock(*a1);
}
