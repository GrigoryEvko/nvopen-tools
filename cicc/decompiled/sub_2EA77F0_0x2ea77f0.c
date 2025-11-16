// Function: sub_2EA77F0
// Address: 0x2ea77f0
//
_QWORD *__fastcall sub_2EA77F0(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // esi
  __int64 v7; // r8
  __int64 v8; // rdi
  int v9; // r11d
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 *v12; // rax
  __int64 v13; // r9
  _QWORD *v14; // rax
  _BYTE *v15; // rsi
  __int64 v16; // rsi
  _QWORD *result; // rax
  int v18; // edi
  int v19; // r8d
  __int64 v20; // r10
  int v21; // edi
  __int64 *v22; // rsi
  int v23; // r8d
  unsigned int v24; // r14d
  __int64 *v25; // rdi
  __int64 v26; // rsi
  unsigned int v27; // r10d
  __int64 v28[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = *(_DWORD *)(a3 + 24);
  if ( !v6 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_36;
  }
  v7 = v6 - 1;
  v8 = *(_QWORD *)(a3 + 8);
  v9 = 1;
  v10 = (unsigned int)v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = v8 + 16 * v10;
  v12 = 0;
  v13 = *(_QWORD *)v11;
  if ( *(_QWORD *)v11 != a2 )
  {
    while ( v13 != -4096 )
    {
      if ( v13 == -8192 && !v12 )
        v12 = (__int64 *)v11;
      v10 = (unsigned int)v7 & (v9 + (_DWORD)v10);
      v11 = v8 + 16LL * (unsigned int)v10;
      v13 = *(_QWORD *)v11;
      if ( *(_QWORD *)v11 == a2 )
        goto LABEL_3;
      ++v9;
    }
    v18 = *(_DWORD *)(a3 + 16);
    if ( !v12 )
      v12 = (__int64 *)v11;
    ++*(_QWORD *)a3;
    v11 = (unsigned int)(v18 + 1);
    if ( 4 * (int)v11 < 3 * v6 )
    {
      v10 = v6 - *(_DWORD *)(a3 + 20) - (unsigned int)v11;
      if ( (unsigned int)v10 > v6 >> 3 )
      {
LABEL_32:
        *(_DWORD *)(a3 + 16) = v11;
        if ( *v12 != -4096 )
          --*(_DWORD *)(a3 + 20);
        *v12 = a2;
        v14 = v12 + 1;
        *v14 = 0;
        goto LABEL_4;
      }
      sub_2EA7610(a3, v6);
      v23 = *(_DWORD *)(a3 + 24);
      if ( v23 )
      {
        v7 = (unsigned int)(v23 - 1);
        v13 = *(_QWORD *)(a3 + 8);
        v10 = 1;
        v24 = v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v11 = (unsigned int)(*(_DWORD *)(a3 + 16) + 1);
        v25 = 0;
        v12 = (__int64 *)(v13 + 16LL * v24);
        v26 = *v12;
        if ( *v12 != a2 )
        {
          while ( v26 != -4096 )
          {
            if ( v26 == -8192 && !v25 )
              v25 = v12;
            v27 = v10 + 1;
            v10 = (unsigned int)v7 & (v24 + (_DWORD)v10);
            v24 = v10;
            v12 = (__int64 *)(v13 + 16LL * (unsigned int)v10);
            v26 = *v12;
            if ( *v12 == a2 )
              goto LABEL_32;
            v10 = v27;
          }
          if ( v25 )
            v12 = v25;
        }
        goto LABEL_32;
      }
LABEL_59:
      ++*(_DWORD *)(a3 + 16);
      BUG();
    }
LABEL_36:
    sub_2EA7610(a3, 2 * v6);
    v19 = *(_DWORD *)(a3 + 24);
    if ( v19 )
    {
      v7 = (unsigned int)(v19 - 1);
      v20 = *(_QWORD *)(a3 + 8);
      v10 = (unsigned int)v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v11 = (unsigned int)(*(_DWORD *)(a3 + 16) + 1);
      v12 = (__int64 *)(v20 + 16 * v10);
      v13 = *v12;
      if ( *v12 != a2 )
      {
        v21 = 1;
        v22 = 0;
        while ( v13 != -4096 )
        {
          if ( !v22 && v13 == -8192 )
            v22 = v12;
          v10 = (unsigned int)v7 & (v21 + (_DWORD)v10);
          v12 = (__int64 *)(v20 + 16LL * (unsigned int)v10);
          v13 = *v12;
          if ( *v12 == a2 )
            goto LABEL_32;
          ++v21;
        }
        if ( v22 )
          v12 = v22;
      }
      goto LABEL_32;
    }
    goto LABEL_59;
  }
LABEL_3:
  v14 = (_QWORD *)(v11 + 8);
LABEL_4:
  *v14 = a1;
  do
  {
    while ( 1 )
    {
      v28[0] = a2;
      v15 = (_BYTE *)a1[5];
      if ( v15 == (_BYTE *)a1[6] )
      {
        sub_2E33A40((__int64)(a1 + 4), v15, v28);
        v16 = v28[0];
      }
      else
      {
        if ( v15 )
        {
          *(_QWORD *)v15 = a2;
          v15 = (_BYTE *)a1[5];
        }
        a1[5] = (__int64)(v15 + 8);
        v16 = a2;
      }
      if ( !*((_BYTE *)a1 + 84) )
        goto LABEL_16;
      result = (_QWORD *)a1[8];
      v10 = *((unsigned int *)a1 + 19);
      v11 = (__int64)&result[v10];
      if ( result != (_QWORD *)v11 )
        break;
LABEL_18:
      if ( (unsigned int)v10 >= *((_DWORD *)a1 + 18) )
      {
LABEL_16:
        result = sub_C8CC70((__int64)(a1 + 7), v16, v11, v10, v7, v13);
        a1 = (__int64 *)*a1;
        if ( !a1 )
          return result;
      }
      else
      {
        v10 = (unsigned int)(v10 + 1);
        *((_DWORD *)a1 + 19) = v10;
        *(_QWORD *)v11 = v16;
        ++a1[7];
        a1 = (__int64 *)*a1;
        if ( !a1 )
          return result;
      }
    }
    while ( v16 != *result )
    {
      if ( (_QWORD *)v11 == ++result )
        goto LABEL_18;
    }
    a1 = (__int64 *)*a1;
  }
  while ( a1 );
  return result;
}
