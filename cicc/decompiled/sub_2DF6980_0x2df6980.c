// Function: sub_2DF6980
// Address: 0x2df6980
//
unsigned __int64 *__fastcall sub_2DF6980(__int64 a1, char a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  int v5; // ebx
  _QWORD *v6; // r13
  unsigned int v7; // r14d
  __int64 v8; // r9
  _QWORD *v9; // rdx
  _QWORD *v10; // rax
  __int64 v11; // rax
  _QWORD *v12; // rdx
  unsigned __int8 v13; // di
  _QWORD *v14; // rsi
  char v15; // al
  _QWORD *v16; // rdi
  char v17; // al
  __int64 v18; // rsi
  __int64 v19; // rsi
  __int64 v20; // rdx
  unsigned __int64 *result; // rax
  __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rax
  const void **v25; // rsi
  void *v26; // r11
  unsigned __int64 v27; // rdi
  size_t v28; // rdx
  unsigned __int64 *v29; // rdx
  unsigned __int64 *v30; // r14
  _QWORD *v31; // rax
  unsigned int v32; // esi
  __int64 v33; // rdi
  unsigned int v34; // edx
  unsigned __int64 *v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+10h] [rbp-50h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  __int64 v39; // [rsp+28h] [rbp-38h]

  v39 = *(_QWORD *)a1;
  v3 = *(_QWORD *)(a1 + 8);
  v4 = v3 + 16LL * *(unsigned int *)(a1 + 16) - 16;
  v5 = *(_DWORD *)(v4 + 8);
  v6 = *(_QWORD **)v4;
  if ( v5 == 1 )
  {
    v30 = v6 + 17;
    do
    {
      if ( *v30 )
        j_j___libc_free_0_0(*v30);
      v30 -= 3;
    }
    while ( v6 + 5 != v30 );
    v31 = *(_QWORD **)(v39 + 168);
    *v6 = *v31;
    *v31 = v6;
    result = (unsigned __int64 *)sub_2DF6700(a1, *(_DWORD *)(v39 + 160));
    if ( a2 )
    {
      if ( *(_DWORD *)(v39 + 160) )
      {
        v32 = *(_DWORD *)(a1 + 16);
        if ( v32 )
        {
          v33 = *(_QWORD *)(a1 + 8);
          v34 = *(_DWORD *)(v33 + 12);
          if ( v34 < *(_DWORD *)(v33 + 8) )
          {
            result = (unsigned __int64 *)(v33 + 28);
            while ( !v34 )
            {
              if ( result == (unsigned __int64 *)(v33 + 28 + 16LL * (v32 - 1)) )
              {
                result = **(unsigned __int64 ***)(v33 + 16LL * v32 - 16);
                *(_QWORD *)v39 = result;
                return result;
              }
              v34 = *(_DWORD *)result;
              result += 2;
            }
          }
        }
      }
    }
  }
  else
  {
    v7 = *(_DWORD *)(v4 + 12) + 1;
    if ( v5 != v7 )
    {
      do
      {
        v8 = v7 - 1;
        v9 = &v6[2 * v7];
        v10 = &v6[2 * v8];
        *v10 = *v9;
        v10[1] = v9[1];
        v11 = 3LL * v7;
        v12 = &v6[3 * v8 + 8];
        if ( &v6[v11 + 8] != v12 )
        {
          if ( (v6[v11 + 9] & 0x3F) != 0 )
          {
            v35 = &v6[3 * v8 + 8];
            v37 = (__int64)&v6[v11 + 8];
            v24 = sub_2207820(4LL * (v6[v11 + 9] & 0x3F));
            v25 = (const void **)v37;
            v8 = v7 - 1;
            v26 = (void *)v24;
            v27 = *v35;
            *v35 = v24;
            if ( v27 )
            {
              j_j___libc_free_0_0(v27);
              v8 = v7 - 1;
              v25 = (const void **)v37;
              v26 = (void *)*v35;
            }
            v13 = v6[3 * v7 + 9] & 0x3F;
            v28 = 4LL * v13;
            if ( v28 )
            {
              v36 = v8;
              memmove(v26, *v25, v28);
              v8 = v36;
              v13 = v6[3 * v7 + 9] & 0x3F;
            }
          }
          else
          {
            *v12 = 0;
            v13 = v6[v11 + 9] & 0x3F;
          }
          v14 = &v6[3 * v8];
          v15 = v13 | v14[9] & 0xC0;
          v16 = &v6[3 * v7];
          *((_BYTE *)v14 + 72) = v15;
          v17 = v16[9] & 0x40 | v15 & 0xBF;
          *((_BYTE *)v14 + 72) = v17;
          *((_BYTE *)v14 + 72) = v16[9] & 0x80 | v17 & 0x7F;
          v14[10] = v16[10];
        }
        ++v7;
      }
      while ( v5 != v7 );
      v3 = *(_QWORD *)(a1 + 8);
      v7 = *(_DWORD *)(v3 + 16LL * *(unsigned int *)(a1 + 16) - 8);
    }
    v18 = *(unsigned int *)(v39 + 160);
    *(_DWORD *)(v3 + 16 * v18 + 8) = v7 - 1;
    if ( (_DWORD)v18 )
    {
      v29 = (unsigned __int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 16LL * (unsigned int)(v18 - 1))
                               + 8LL * *(unsigned int *)(*(_QWORD *)(a1 + 8) + 16LL * (unsigned int)(v18 - 1) + 12));
      *v29 = (v7 - 2) | *v29 & 0xFFFFFFFFFFFFFFC0LL;
    }
    v19 = *(_QWORD *)(a1 + 8);
    v20 = *(unsigned int *)(a1 + 16);
    result = (unsigned __int64 *)(16 * v20);
    v22 = v19 + 16 * v20 - 16;
    if ( *(_DWORD *)(v22 + 12) == v7 - 1 )
    {
      sub_2DF4670(a1, *(_DWORD *)(v39 + 160), v6[2 * v7 - 3]);
      return sub_F03D40((__int64 *)(a1 + 8), *(_DWORD *)(v39 + 160));
    }
    else if ( a2 )
    {
      if ( (_DWORD)v20 )
      {
        result = (unsigned __int64 *)(v19 + 12);
        v23 = v19 + 16LL * (unsigned int)(v20 - 1) + 28;
        while ( !*(_DWORD *)result )
        {
          result += 2;
          if ( (unsigned __int64 *)v23 == result )
            goto LABEL_36;
        }
      }
      else
      {
LABEL_36:
        result = **(unsigned __int64 ***)v22;
        *(_QWORD *)v39 = result;
      }
    }
  }
  return result;
}
