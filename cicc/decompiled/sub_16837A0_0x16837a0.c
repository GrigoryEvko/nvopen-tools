// Function: sub_16837A0
// Address: 0x16837a0
//
__int64 __fastcall sub_16837A0(__int64 a1)
{
  __int64 v1; // r12
  unsigned int v2; // eax
  bool v3; // zf
  unsigned int v4; // r14d
  pthread_mutex_t *v6; // r8
  _BYTE *v7; // rsi
  ssize_t v8; // rax
  int *v9; // rdx
  pthread_mutex_t *v10; // [rsp+8h] [rbp-48h]
  pthread_mutex_t *mutex; // [rsp+10h] [rbp-40h] BYREF
  char v12; // [rsp+18h] [rbp-38h]

  v1 = a1 + 136;
LABEL_2:
  if ( *(_DWORD *)a1 )
    return (unsigned int)*__errno_location();
  while ( 1 )
  {
    if ( *(_BYTE *)(a1 + 205) )
      return (unsigned int)*__errno_location();
    mutex = (pthread_mutex_t *)(a1 + 56);
    v12 = 0;
    if ( &_pthread_key_create )
    {
      v2 = pthread_mutex_lock((pthread_mutex_t *)(a1 + 56));
      if ( v2 )
        goto LABEL_51;
    }
    v3 = *(_QWORD *)(a1 + 16) == 0;
    v12 = 1;
    if ( v3 )
    {
      do
      {
        if ( *(_BYTE *)(a1 + 205) )
          break;
        sub_2210B30(v1, &mutex);
      }
      while ( !*(_QWORD *)(a1 + 16) );
      if ( !v12 )
LABEL_23:
        sub_4264C5(1u);
    }
    if ( mutex )
    {
      if ( &_pthread_key_create )
        pthread_mutex_unlock(mutex);
      v12 = 0;
    }
    if ( *(_BYTE *)(a1 + 185) )
    {
      _InterlockedCompareExchange((volatile signed __int32 *)a1, 2, 0);
      v4 = *__errno_location();
      if ( !v12 )
        return v4;
      goto LABEL_44;
    }
    if ( !sub_16825F0(a1, (void *)(a1 + 185)) )
    {
      v4 = *__errno_location();
      if ( v4 != 11 )
        goto LABEL_43;
      goto LABEL_14;
    }
    if ( !mutex )
      goto LABEL_23;
    if ( v12 )
      sub_4264C5(0x23u);
    if ( &_pthread_key_create )
    {
      v2 = pthread_mutex_lock(mutex);
      if ( v2 )
LABEL_51:
        sub_4264C5(v2);
    }
    v3 = *(_QWORD *)(a1 + 16) == 0;
    v12 = 1;
    if ( !v3 )
    {
      *(_BYTE *)(a1 + 184) = 1;
      v6 = (pthread_mutex_t *)(a1 + 96);
      if ( &_pthread_key_create )
      {
        v2 = pthread_mutex_lock((pthread_mutex_t *)(a1 + 96));
        v6 = (pthread_mutex_t *)(a1 + 96);
        if ( v2 )
          goto LABEL_51;
      }
      v7 = *(_BYTE **)(a1 + 40);
      if ( v7 == *(_BYTE **)(a1 + 48) )
      {
        v10 = v6;
        sub_1683630(a1 + 32, v7, (char *)(a1 + 185));
        v6 = v10;
      }
      else
      {
        if ( v7 )
        {
          *v7 = *(_BYTE *)(a1 + 185);
          v7 = *(_BYTE **)(a1 + 40);
        }
        *(_QWORD *)(a1 + 40) = v7 + 1;
      }
      *(_BYTE *)(a1 + 185) = 0;
      if ( &_pthread_key_create )
        pthread_mutex_unlock(v6);
      sub_2210B70(v1);
      goto LABEL_14;
    }
    v8 = write(*(_DWORD *)(a1 + 192), (const void *)(a1 + 185), 1u);
    *(_BYTE *)(a1 + 185) = 0;
    if ( v8 == -1 )
    {
      v9 = __errno_location();
      *(_DWORD *)(a1 + 4) = *v9;
      _InterlockedCompareExchange((volatile signed __int32 *)a1, 11, 0);
      goto LABEL_42;
    }
    if ( v8 != 1 )
      break;
LABEL_14:
    if ( !v12 || !mutex || !&_pthread_key_create )
      goto LABEL_2;
    pthread_mutex_unlock(mutex);
    if ( *(_DWORD *)a1 )
      return (unsigned int)*__errno_location();
  }
  v9 = __errno_location();
LABEL_42:
  v4 = *v9;
LABEL_43:
  if ( v12 )
  {
LABEL_44:
    if ( mutex && &_pthread_key_create )
      pthread_mutex_unlock(mutex);
  }
  return v4;
}
