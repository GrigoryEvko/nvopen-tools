// Function: sub_16827A0
// Address: 0x16827a0
//
__int64 __fastcall sub_16827A0(__int64 a1)
{
  pthread_mutex_t *v2; // r12
  unsigned int v3; // eax
  unsigned int v4; // r15d
  pthread_mutex_t *v6; // rdi
  bool v7; // zf
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  pthread_mutex_t *mutex; // [rsp+10h] [rbp-40h] BYREF
  char v11; // [rsp+18h] [rbp-38h]

  if ( !a1 )
    return 4;
  v2 = (pthread_mutex_t *)(a1 + 216);
  if ( &_pthread_key_create )
  {
    v3 = pthread_mutex_lock((pthread_mutex_t *)(a1 + 216));
    if ( v3 )
      goto LABEL_29;
  }
  v4 = *(_DWORD *)a1;
  if ( *(_DWORD *)a1 )
    goto LABEL_5;
  v6 = (pthread_mutex_t *)(a1 + 56);
  v11 = 0;
  mutex = (pthread_mutex_t *)(a1 + 56);
  if ( &_pthread_key_create )
  {
    v3 = pthread_mutex_lock(v6);
    if ( v3 )
LABEL_29:
      sub_4264C5(v3);
  }
  v7 = *(_BYTE *)(a1 + 8) == 0;
  v11 = 1;
  if ( v7 )
  {
    v8 = *(_QWORD *)(a1 + 16) + 1LL;
    *(_QWORD *)(a1 + 16) = v8;
    if ( v8 > 1 )
    {
      v4 = 2;
    }
    else
    {
      sub_2210B70(a1 + 136);
      if ( !*(_BYTE *)(a1 + 184) )
      {
        while ( !*(_BYTE *)(a1 + 205) )
        {
          sub_2210B30(a1 + 136, &mutex);
          if ( *(_BYTE *)(a1 + 184) )
          {
            if ( !*(_BYTE *)(a1 + 205) )
              goto LABEL_14;
            goto LABEL_21;
          }
        }
      }
      if ( *(_BYTE *)(a1 + 205) )
      {
LABEL_21:
        v4 = 3;
      }
      else
      {
LABEL_14:
        v9 = *(_QWORD *)(a1 + 16);
        *(_BYTE *)(a1 + 184) = 0;
        if ( v9 )
        {
          ++*(_QWORD *)(a1 + 24);
          *(_QWORD *)(a1 + 16) = v9 - 1;
        }
        else
        {
          v4 = 2;
        }
      }
      if ( !v11 )
        goto LABEL_5;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = 0;
  }
  if ( mutex )
  {
    if ( !&_pthread_key_create )
      return v4;
    pthread_mutex_unlock(mutex);
  }
LABEL_5:
  if ( &_pthread_key_create )
    pthread_mutex_unlock(v2);
  return v4;
}
