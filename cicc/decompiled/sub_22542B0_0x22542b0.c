// Function: sub_22542B0
// Address: 0x22542b0
//
int __fastcall sub_22542B0(pthread_mutex_t *mutex, int a2)
{
  char *v4; // rdi
  char *v5; // rbp
  __int64 v6; // rax
  __int64 v7; // rdx
  int **v8; // rcx
  _DWORD *v9; // r14
  char *v10; // rdx
  char *v11; // rsi
  int lock; // eax

  if ( &_pthread_key_create && pthread_mutex_lock(mutex) )
    JUMPOUT(0x4267BE);
  v4 = (char *)*(&mutex[1].__align + 2);
  v5 = (char *)*(&mutex[1].__align + 1);
  v6 = (*(&mutex[1].__align + 2) - (__int64)v5) >> 3;
  if ( *(&mutex[1].__align + 2) - (__int64)v5 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v7 = v6 >> 1;
        v8 = (int **)&v5[8 * (v6 >> 1)];
        if ( **v8 >= a2 )
          break;
        v5 = (char *)(v8 + 1);
        v6 = v6 - v7 - 1;
        if ( v6 <= 0 )
          goto LABEL_8;
      }
      v6 >>= 1;
    }
    while ( v7 > 0 );
  }
LABEL_8:
  if ( v4 != v5 && (v9 = *(_DWORD **)v5, **(_DWORD **)v5 == a2) )
  {
    _libc_free(*((_QWORD *)v9 + 1));
    sub_2209150((volatile signed __int32 **)v9 + 2);
    j___libc_free_0((unsigned __int64)v9);
    v10 = (char *)*(&mutex[1].__align + 2);
    v11 = v5 + 8;
    if ( v10 != v5 + 8 )
    {
      memmove(v5, v11, v10 - v11);
      v11 = (char *)*(&mutex[1].__align + 2);
    }
    lock = mutex[1].__lock;
    *(&mutex[1].__align + 2) = (__int64)(v11 - 8);
    LODWORD(v6) = lock - 1;
    if ( (_DWORD)v6 == a2 )
      mutex[1].__lock = a2;
    if ( &_pthread_key_create )
    {
      LODWORD(v6) = pthread_mutex_unlock(mutex);
      if ( (_DWORD)v6 )
        JUMPOUT(0x426790);
    }
  }
  else if ( &_pthread_key_create )
  {
    LODWORD(v6) = pthread_mutex_unlock(mutex);
    if ( (_DWORD)v6 )
      JUMPOUT(0x4267C3);
  }
  return v6;
}
