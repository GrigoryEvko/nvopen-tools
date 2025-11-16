// Function: sub_29214F0
// Address: 0x29214f0
//
__int64 *__fastcall sub_29214F0(__int64 a1, __int64 a2)
{
  __int64 *result; // rax
  __int64 v4; // rdx
  __int64 *v5; // r13
  __int64 v6; // rdx
  __int64 *v7; // rbx
  int v8; // eax
  _QWORD *v9; // rdi
  __int64 v10; // rsi
  _QWORD *v11; // rax
  int v12; // r8d
  const void *v13; // r9
  int v14; // eax
  int v15; // eax
  __int64 v16; // rdi
  int v17; // ecx
  unsigned int v18; // eax
  __int64 *v19; // rsi
  __int64 v20; // r8
  __int64 v21; // rax
  _QWORD *v22; // rdi
  int v23; // esi
  int v24; // r9d
  __int64 v25[5]; // [rsp+8h] [rbp-28h] BYREF

  result = *(__int64 **)(a2 + 8);
  if ( *(_BYTE *)(a2 + 28) )
    v4 = *(unsigned int *)(a2 + 20);
  else
    v4 = *(unsigned int *)(a2 + 16);
  v5 = &result[v4];
  if ( result != v5 )
  {
    while ( 1 )
    {
      v6 = *result;
      v7 = result;
      if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v5 == ++result )
        return result;
    }
    if ( v5 != result )
    {
      v8 = *(_DWORD *)(a1 + 16);
      v25[0] = v6;
      if ( v8 )
        goto LABEL_19;
LABEL_9:
      v9 = *(_QWORD **)(a1 + 32);
      v10 = (__int64)&v9[*(unsigned int *)(a1 + 40)];
      v11 = sub_2912630(v9, v10, v25);
      if ( (_QWORD *)v10 == v11 )
        goto LABEL_13;
      v13 = v11 + 1;
      if ( (_QWORD *)v10 != v11 + 1 )
      {
LABEL_11:
        memmove(v11, v13, v10 - (_QWORD)v13);
        v12 = *(_DWORD *)(a1 + 40);
      }
LABEL_12:
      *(_DWORD *)(a1 + 40) = v12 - 1;
LABEL_13:
      while ( 1 )
      {
        result = v7 + 1;
        if ( v7 + 1 == v5 )
          break;
        v6 = *result;
        for ( ++v7; (unsigned __int64)*result >= 0xFFFFFFFFFFFFFFFELL; v7 = result )
        {
          if ( v5 == ++result )
            return result;
          v6 = *result;
        }
        if ( v5 == v7 )
          return result;
        v14 = *(_DWORD *)(a1 + 16);
        v25[0] = v6;
        if ( !v14 )
          goto LABEL_9;
LABEL_19:
        v15 = *(_DWORD *)(a1 + 24);
        v16 = *(_QWORD *)(a1 + 8);
        if ( v15 )
        {
          v17 = v15 - 1;
          v18 = (v15 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
          v19 = (__int64 *)(v16 + 8LL * v18);
          v20 = *v19;
          if ( *v19 == v6 )
          {
LABEL_21:
            *v19 = -8192;
            v21 = *(unsigned int *)(a1 + 40);
            --*(_DWORD *)(a1 + 16);
            v22 = *(_QWORD **)(a1 + 32);
            ++*(_DWORD *)(a1 + 20);
            v10 = (__int64)&v22[v21];
            v11 = sub_2912630(v22, v10, v25);
            v13 = v11 + 1;
            if ( v11 + 1 != (_QWORD *)v10 )
              goto LABEL_11;
            goto LABEL_12;
          }
          v23 = 1;
          while ( v20 != -4096 )
          {
            v24 = v23 + 1;
            v18 = v17 & (v23 + v18);
            v19 = (__int64 *)(v16 + 8LL * v18);
            v20 = *v19;
            if ( *v19 == v6 )
              goto LABEL_21;
            v23 = v24;
          }
        }
      }
    }
  }
  return result;
}
