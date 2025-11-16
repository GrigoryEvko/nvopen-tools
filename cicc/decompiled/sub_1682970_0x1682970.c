// Function: sub_1682970
// Address: 0x1682970
//
__int64 __fastcall sub_1682970(__int64 a1)
{
  pthread_mutex_t *v2; // r14
  unsigned int v3; // eax
  unsigned int v4; // r13d
  __int64 v6; // rax
  int v7; // edi
  ssize_t v8; // rax
  __int64 v9; // rax
  pthread_mutex_t *mutex; // [rsp+8h] [rbp-48h]
  _BYTE buf[49]; // [rsp+1Fh] [rbp-31h] BYREF

  if ( !a1 )
    return 4;
  v2 = (pthread_mutex_t *)(a1 + 256);
  if ( &_pthread_key_create )
  {
    v3 = pthread_mutex_lock((pthread_mutex_t *)(a1 + 256));
    if ( v3 )
      goto LABEL_34;
  }
  v4 = *(_DWORD *)a1;
  if ( *(_DWORD *)a1 )
    goto LABEL_5;
  if ( (mutex = (pthread_mutex_t *)(a1 + 56), &_pthread_key_create) && (v3 = pthread_mutex_lock(mutex)) != 0
    || &_pthread_key_create && (v3 = pthread_mutex_lock((pthread_mutex_t *)(a1 + 96))) != 0 )
  {
LABEL_34:
    sub_4264C5(v3);
  }
  v6 = *(_QWORD *)(a1 + 40);
  if ( v6 != *(_QWORD *)(a1 + 32) )
  {
    v7 = *(_DWORD *)(a1 + 192);
    buf[0] = *(_BYTE *)(v6 - 1);
    v8 = write(v7, buf, 1u);
    if ( v8 != 1 )
    {
      if ( v8 == -1 )
      {
        *(_DWORD *)(a1 + 4) = *__errno_location();
        _InterlockedCompareExchange((volatile signed __int32 *)a1, 11, 0);
      }
      else
      {
        _InterlockedCompareExchange((volatile signed __int32 *)a1, 2, 0);
      }
LABEL_16:
      if ( &_pthread_key_create )
        pthread_mutex_unlock((pthread_mutex_t *)(a1 + 96));
      goto LABEL_18;
    }
    --*(_QWORD *)(a1 + 40);
    if ( &_pthread_key_create )
      pthread_mutex_unlock((pthread_mutex_t *)(a1 + 96));
    --*(_QWORD *)(a1 + 24);
LABEL_19:
    if ( !&_pthread_key_create )
      return v4;
    goto LABEL_20;
  }
  if ( *(_BYTE *)(a1 + 8) )
    goto LABEL_16;
  if ( &_pthread_key_create )
    pthread_mutex_unlock((pthread_mutex_t *)(a1 + 96));
  v9 = *(_QWORD *)(a1 + 24);
  if ( !*(_QWORD *)(a1 + 16) )
  {
    *(_BYTE *)(a1 + 8) = 1;
    if ( v9 != 1 )
    {
LABEL_18:
      _InterlockedCompareExchange((volatile signed __int32 *)a1, 12, 0);
      v4 = 12;
      goto LABEL_19;
    }
    *(_QWORD *)(a1 + 24) = 0;
    goto LABEL_19;
  }
  *(_BYTE *)(a1 + 184) = 1;
  *(_QWORD *)(a1 + 24) = v9 - 1;
  sub_2210B70(a1 + 136);
  if ( !&_pthread_key_create )
    return v4;
LABEL_20:
  pthread_mutex_unlock(mutex);
LABEL_5:
  if ( &_pthread_key_create )
    pthread_mutex_unlock(v2);
  return v4;
}
