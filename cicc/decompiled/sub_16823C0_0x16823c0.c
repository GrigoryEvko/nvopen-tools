// Function: sub_16823C0
// Address: 0x16823c0
//
int __fastcall sub_16823C0(__int64 a1, char *p_buf)
{
  signed __int8 v3; // al
  unsigned int v4; // eax
  __int64 v5; // rdx
  int v6; // edi
  __int64 v7; // rax
  int v8; // edi
  ssize_t v9; // rax
  int *v10; // rax
  int v11; // edi
  __int64 v12; // rdi
  int result; // eax
  __int64 v14; // rdi
  int v15; // edi
  int v16; // r8d
  char buf; // [rsp+Eh] [rbp-32h] BYREF
  _BYTE v18[49]; // [rsp+Fh] [rbp-31h] BYREF

  v18[0] = 0;
  v3 = _InterlockedCompareExchange8((volatile signed __int8 *)(a1 + 205), 1, 0);
  if ( v3 )
    v18[0] = v3;
  if ( &_pthread_key_create )
  {
    v4 = pthread_mutex_lock((pthread_mutex_t *)(a1 + 56));
    if ( v4 )
      goto LABEL_37;
  }
  sub_2210B70(a1 + 136);
  if ( &_pthread_key_create )
    pthread_mutex_unlock((pthread_mutex_t *)(a1 + 56));
  v6 = *(_DWORD *)(a1 + 200);
  buf = 0;
  if ( v6 >= 0 )
  {
    p_buf = &buf;
    write(v6, &buf, 1u);
  }
  if ( *(_QWORD *)(a1 + 208) )
    sub_2242090(a1 + 208);
  *(_BYTE *)(a1 + 8) = 1;
  if ( &_pthread_key_create )
  {
    v4 = pthread_mutex_lock((pthread_mutex_t *)(a1 + 56));
    if ( v4 )
LABEL_37:
      sub_4264C5(v4);
  }
  v7 = *(_QWORD *)(a1 + 40);
  if ( v7 != *(_QWORD *)(a1 + 32) )
  {
    while ( 1 )
    {
      v8 = *(_DWORD *)(a1 + 192);
      p_buf = v18;
      v18[0] = *(_BYTE *)(v7 - 1);
      v9 = write(v8, v18, 1u);
      if ( v9 == 1 )
        break;
      if ( v9 == -1 )
      {
        v10 = __errno_location();
        v5 = 11;
        *(_DWORD *)(a1 + 4) = *v10;
        _InterlockedCompareExchange((volatile signed __int32 *)a1, 11, 0);
        v7 = *(_QWORD *)(a1 + 40);
        if ( *(_QWORD *)(a1 + 32) == v7 )
          goto LABEL_20;
      }
      else
      {
        v5 = 2;
        _InterlockedCompareExchange((volatile signed __int32 *)a1, 2, 0);
        v7 = *(_QWORD *)(a1 + 40);
LABEL_16:
        if ( *(_QWORD *)(a1 + 32) == v7 )
          goto LABEL_20;
      }
    }
    v7 = *(_QWORD *)(a1 + 40) - 1LL;
    *(_QWORD *)(a1 + 40) = v7;
    goto LABEL_16;
  }
LABEL_20:
  if ( &_pthread_key_create )
    pthread_mutex_unlock((pthread_mutex_t *)(a1 + 56));
  if ( *(_BYTE *)(a1 + 204) )
  {
    v15 = *(_DWORD *)(a1 + 188);
    if ( v15 >= 0 )
    {
      close(v15);
      v15 = *(_DWORD *)(a1 + 188);
    }
    v16 = *(_DWORD *)(a1 + 192);
    if ( v16 >= 0 && v16 != v15 )
      close(v16);
  }
  v11 = *(_DWORD *)(a1 + 196);
  if ( v11 >= 0 )
    close(v11);
  v12 = *(unsigned int *)(a1 + 200);
  if ( (int)v12 >= 0 )
    close(v12);
  if ( *(_QWORD *)(a1 + 208) )
    sub_2207530(v12, p_buf, v5);
  result = j__pthread_cond_destroy((pthread_cond_t *)(a1 + 136));
  v14 = *(_QWORD *)(a1 + 32);
  if ( v14 )
    return j_j___libc_free_0(v14, *(_QWORD *)(a1 + 48) - v14);
  return result;
}
