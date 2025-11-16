// Function: sub_CA0000
// Address: 0xca0000
//
__int64 __fastcall sub_CA0000(__int64 a1, _QWORD *a2, char a3)
{
  pthread_mutex_t *v4; // r13
  unsigned int v5; // eax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 result; // rax

  if ( !qword_4F84F60 )
    sub_C7D570(&qword_4F84F60, sub_CA0780, (__int64)sub_C9FD10);
  v4 = (pthread_mutex_t *)(qword_4F84F60 + 664);
  if ( &_pthread_key_create )
  {
    v5 = pthread_mutex_lock((pthread_mutex_t *)(qword_4F84F60 + 664));
    if ( v5 )
      sub_4264C5(v5);
  }
  sub_C9FF30(a1, a3);
  if ( &_pthread_key_create )
    pthread_mutex_unlock(v4);
  result = *(_QWORD *)(a1 + 72);
  if ( *(_QWORD *)(a1 + 80) != result )
    return sub_C9EA90(a1, a2, v6, v7, v8);
  return result;
}
