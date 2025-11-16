// Function: sub_16D4EC0
// Address: 0x16d4ec0
//
int __fastcall sub_16D4EC0(__int64 a1)
{
  pthread_mutex_t *v2; // rdi
  unsigned int v3; // eax
  __int64 v4; // rax
  pthread_mutex_t *mutex; // [rsp+0h] [rbp-40h] BYREF
  char v7; // [rsp+8h] [rbp-38h]

  v2 = (pthread_mutex_t *)(a1 + 192);
  mutex = v2;
  v7 = 0;
  if ( &_pthread_key_create )
  {
    v3 = pthread_mutex_lock(v2);
    if ( v3 )
      sub_4264C5(v3);
  }
  v7 = 1;
  while ( 1 )
  {
    if ( !*(_DWORD *)(a1 + 280) )
    {
      v4 = *(_QWORD *)(a1 + 40);
      if ( *(_QWORD *)(a1 + 72) == v4 )
        break;
    }
    sub_2210B30(a1 + 232, &mutex);
  }
  if ( v7 && mutex && &_pthread_key_create )
    LODWORD(v4) = pthread_mutex_unlock(mutex);
  return v4;
}
