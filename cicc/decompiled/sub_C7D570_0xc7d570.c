// Function: sub_C7D570
// Address: 0xc7d570
//
int __fastcall sub_C7D570(__int64 *a1, __int64 (*a2)(void), __int64 a3)
{
  unsigned int v4; // eax
  __int64 v5; // rax

  if ( &_pthread_key_create )
  {
    v4 = pthread_mutex_lock(&stru_4C5C740);
    if ( v4 )
      sub_4264C5(v4);
  }
  v5 = *a1;
  if ( *a1 )
  {
    if ( !&_pthread_key_create )
      return v5;
  }
  else
  {
    *a1 = a2();
    a1[1] = a3;
    v5 = qword_4F840F0;
    qword_4F840F0 = (__int64)a1;
    a1[2] = v5;
    if ( !&_pthread_key_create )
      return v5;
  }
  LODWORD(v5) = pthread_mutex_unlock(&stru_4C5C740);
  return v5;
}
