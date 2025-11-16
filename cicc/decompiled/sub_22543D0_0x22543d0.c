// Function: sub_22543D0
// Address: 0x22543d0
//
_DWORD *__fastcall sub_22543D0(pthread_mutex_t *mutex, int a2)
{
  _QWORD *v3; // r8
  _QWORD *v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rdx
  _DWORD **v7; // rcx
  _DWORD *v8; // r12

  if ( &_pthread_key_create && pthread_mutex_lock(mutex) )
    JUMPOUT(0x42684C);
  v3 = (_QWORD *)*(&mutex[1].__align + 2);
  v4 = (_QWORD *)*(&mutex[1].__align + 1);
  v5 = (*(&mutex[1].__align + 2) - (__int64)v4) >> 3;
  if ( *(&mutex[1].__align + 2) - (__int64)v4 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v6 = v5 >> 1;
        v7 = (_DWORD **)&v4[v5 >> 1];
        if ( **v7 >= a2 )
          break;
        v4 = v7 + 1;
        v5 = v5 - v6 - 1;
        if ( v5 <= 0 )
          goto LABEL_8;
      }
      v5 >>= 1;
    }
    while ( v6 > 0 );
  }
LABEL_8:
  if ( v3 == v4 )
  {
    v8 = 0;
  }
  else
  {
    v8 = (_DWORD *)*v4;
    if ( *(_DWORD *)*v4 != a2 )
      v8 = 0;
  }
  if ( &_pthread_key_create && pthread_mutex_unlock(mutex) )
    JUMPOUT(0x42681E);
  return v8;
}
