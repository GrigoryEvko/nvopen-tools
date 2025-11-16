// Function: sub_D3F9D0
// Address: 0xd3f9d0
//
char *__fastcall sub_D3F9D0(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v7; // rsi
  __int64 v8; // r8
  __int64 v9; // r9
  char *result; // rax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rcx
  int v13; // r13d
  char *v14; // rdi
  __int64 v15; // rdx
  __int64 *v16; // rcx
  char *v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 *v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // rsi
  char *v24; // [rsp+0h] [rbp-70h] BYREF
  char *v25; // [rsp+8h] [rbp-68h]
  _BYTE src[96]; // [rsp+10h] [rbp-60h] BYREF

  sub_D3F850(a1, a2, a3, a4, a5, a6);
  v7 = (char *)a1;
  sub_D36170((__int64)&v24, a1);
  result = v24;
  if ( v24 == src )
  {
    v11 = (unsigned int)v25;
    v12 = *(unsigned int *)(a1 + 304);
    v13 = (int)v25;
    if ( (unsigned int)v25 <= v12 )
    {
      v14 = src;
      if ( !(_DWORD)v25 )
      {
LABEL_8:
        *(_DWORD *)(a1 + 304) = v13;
        if ( v14 != src )
          return (char *)_libc_free(v14, v7);
        return result;
      }
      result = *(char **)(a1 + 296);
      v16 = (__int64 *)src;
      v17 = &result[16 * (unsigned int)v25];
      do
      {
        v18 = *v16;
        result += 16;
        v16 += 2;
        *((_QWORD *)result - 2) = v18;
        v7 = (char *)*(v16 - 1);
        *((_QWORD *)result - 1) = v7;
      }
      while ( result != v17 );
    }
    else
    {
      if ( (unsigned int)v25 > (unsigned __int64)*(unsigned int *)(a1 + 308) )
      {
        *(_DWORD *)(a1 + 304) = 0;
        sub_C8D5F0(a1 + 296, (const void *)(a1 + 312), v11, 0x10u, v8, v9);
        v14 = v24;
        v11 = (unsigned int)v25;
        v12 = 0;
        v7 = v24;
      }
      else
      {
        v7 = src;
        v14 = src;
        if ( *(_DWORD *)(a1 + 304) )
        {
          v20 = *(_QWORD *)(a1 + 296);
          v12 *= 16LL;
          v21 = (__int64 *)src;
          v22 = v20 + v12;
          do
          {
            v23 = *v21;
            v20 += 16;
            v21 += 2;
            *(_QWORD *)(v20 - 16) = v23;
            *(_QWORD *)(v20 - 8) = *(v21 - 1);
          }
          while ( v20 != v22 );
          v14 = v24;
          v11 = (unsigned int)v25;
          v7 = &v24[v12];
        }
      }
      v15 = 16 * v11;
      result = &v14[v15];
      if ( v7 == &v14[v15] )
        goto LABEL_8;
      result = (char *)memcpy((void *)(v12 + *(_QWORD *)(a1 + 296)), v7, v15 - v12);
    }
    v14 = v24;
    goto LABEL_8;
  }
  v19 = *(_QWORD *)(a1 + 296);
  if ( v19 != a1 + 312 )
  {
    _libc_free(v19, v7);
    result = v24;
  }
  *(_QWORD *)(a1 + 296) = result;
  result = v25;
  *(_QWORD *)(a1 + 304) = v25;
  return result;
}
