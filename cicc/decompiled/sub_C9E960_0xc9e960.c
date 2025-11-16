// Function: sub_C9E960
// Address: 0xc9e960
//
int __fastcall sub_C9E960(__int64 a1, __int64 a2)
{
  pthread_mutex_t *v2; // r13
  unsigned int v3; // eax
  __int64 v4; // rax
  int result; // eax

  if ( !qword_4F84F60 )
    sub_C7D570(&qword_4F84F60, sub_CA0780, (__int64)sub_C9FD10);
  v2 = (pthread_mutex_t *)(qword_4F84F60 + 664);
  if ( &_pthread_key_create )
  {
    v3 = pthread_mutex_lock((pthread_mutex_t *)(qword_4F84F60 + 664));
    if ( v3 )
      sub_4264C5(v3);
  }
  v4 = *(_QWORD *)(a1 + 64);
  if ( v4 )
  {
    *(_QWORD *)(v4 + 160) = a2 + 168;
    v4 = *(_QWORD *)(a1 + 64);
  }
  *(_QWORD *)(a2 + 168) = v4;
  result = a1 + 64;
  *(_QWORD *)(a2 + 160) = a1 + 64;
  *(_QWORD *)(a1 + 64) = a2;
  if ( &_pthread_key_create )
    return pthread_mutex_unlock(v2);
  return result;
}
