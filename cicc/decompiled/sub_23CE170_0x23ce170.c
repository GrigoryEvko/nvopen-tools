// Function: sub_23CE170
// Address: 0x23ce170
//
int __fastcall sub_23CE170(__int64 a1, __int64 a2)
{
  int result; // eax
  pthread_mutex_t *v4; // rdi
  unsigned int v5; // eax
  pthread_mutex_t *mutex; // [rsp+0h] [rbp-40h] BYREF
  char v7; // [rsp+8h] [rbp-38h]

  if ( (unsigned __int8)sub_23CCE00(a1) )
    return sub_23CDA10(a1, a2);
  v4 = (pthread_mutex_t *)(a1 + 176);
  v7 = 0;
  mutex = (pthread_mutex_t *)(a1 + 176);
  if ( &_pthread_key_create )
  {
    v5 = pthread_mutex_lock(v4);
    if ( v5 )
      sub_4264C5(v5);
  }
  v7 = 1;
  while ( 1 )
  {
    result = sub_23CCCA0(a1, a2);
    if ( (_BYTE)result )
      break;
    sub_2210B30((pthread_cond_t *)(a1 + 264), &mutex);
  }
  if ( v7 && mutex )
  {
    if ( &_pthread_key_create )
      return pthread_mutex_unlock(mutex);
  }
  return result;
}
