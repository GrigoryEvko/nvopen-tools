// Function: sub_23CCD50
// Address: 0x23ccd50
//
int __fastcall sub_23CCD50(__int64 a1)
{
  pthread_mutex_t *v2; // rdi
  unsigned int v3; // eax
  int result; // eax
  pthread_mutex_t *mutex; // [rsp+0h] [rbp-30h] BYREF
  char v6; // [rsp+8h] [rbp-28h]

  v2 = (pthread_mutex_t *)(a1 + 176);
  mutex = v2;
  v6 = 0;
  if ( &_pthread_key_create )
  {
    v3 = pthread_mutex_lock(v2);
    if ( v3 )
      sub_4264C5(v3);
  }
  v6 = 1;
  while ( 1 )
  {
    result = sub_23CCCA0(a1, 0);
    if ( (_BYTE)result )
      break;
    sub_2210B30((pthread_cond_t *)(a1 + 264), &mutex);
  }
  if ( v6 && mutex )
  {
    if ( &_pthread_key_create )
      return pthread_mutex_unlock(mutex);
  }
  return result;
}
