// Function: sub_22A9FC0
// Address: 0x22a9fc0
//
__int64 *__fastcall sub_22A9FC0(__int64 a1, __int64 a2)
{
  unsigned int v3; // esi
  __int64 v4; // r9
  unsigned int v5; // r8d
  unsigned int v6; // ebx
  unsigned int v7; // ecx
  __int64 *v8; // rax
  __int64 v9; // rdx
  int v11; // eax
  int v12; // eax
  __int64 v13; // r8
  unsigned int v14; // ecx
  __int64 *v15; // r10
  __int64 v16; // rsi
  int v17; // edx
  int v18; // r11d
  __int64 *v19; // r9
  __int64 v20; // r10
  __int64 v21; // r13
  int i; // r11d
  int v23; // r11d
  int v24; // eax
  int v25; // r14d
  __int64 *v26; // r11
  int v27; // eax
  int v28; // eax
  __int64 v29; // rsi
  int v30; // r9d
  __int64 *v31; // r8
  unsigned int v32; // ebx
  __int64 v33; // rcx
  _QWORD *v34; // [rsp+8h] [rbp-28h]

  v3 = *(_DWORD *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  if ( !v3 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_5;
  }
  v5 = v3 - 1;
  v6 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v7 = (v3 - 1) & v6;
  v8 = (__int64 *)(v4 + 24LL * v7);
  v9 = *v8;
  if ( *v8 == a2 )
    return v8 + 1;
  v20 = *v8;
  LODWORD(v21) = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  for ( i = 1; ; i = v25 )
  {
    if ( v20 == -4096 )
    {
      v6 = ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9);
      v7 = v6 & v5;
      v8 = (__int64 *)(v4 + 24LL * (v6 & v5));
      v9 = *v8;
      if ( *v8 == a2 )
        return v8 + 1;
LABEL_15:
      v23 = 1;
      v15 = 0;
      while ( v9 != -4096 )
      {
        if ( v9 == -8192 && !v15 )
          v15 = v8;
        v7 = v5 & (v23 + v7);
        v8 = (__int64 *)(v4 + 24LL * v7);
        v9 = *v8;
        if ( *v8 == a2 )
          return v8 + 1;
        ++v23;
      }
      if ( !v15 )
        v15 = v8;
      v24 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)a1;
      v17 = v24 + 1;
      if ( 4 * (v24 + 1) < 3 * v3 )
      {
        if ( v3 - *(_DWORD *)(a1 + 20) - v17 > v3 >> 3 )
        {
LABEL_21:
          *(_DWORD *)(a1 + 16) = v17;
          if ( *v15 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v15 = a2;
          v34 = v15 + 1;
          sub_22A6980((__int64)(v15 + 1), a2, 0, 0, 0, 0);
          return v34;
        }
        sub_22A9DC0(a1, v3);
        v27 = *(_DWORD *)(a1 + 24);
        if ( v27 )
        {
          v28 = v27 - 1;
          v29 = *(_QWORD *)(a1 + 8);
          v30 = 1;
          v31 = 0;
          v32 = v28 & v6;
          v15 = (__int64 *)(v29 + 24LL * v32);
          v33 = *v15;
          v17 = *(_DWORD *)(a1 + 16) + 1;
          if ( *v15 != a2 )
          {
            while ( v33 != -4096 )
            {
              if ( !v31 && v33 == -8192 )
                v31 = v15;
              v32 = v28 & (v30 + v32);
              v15 = (__int64 *)(v29 + 24LL * v32);
              v33 = *v15;
              if ( *v15 == a2 )
                goto LABEL_21;
              ++v30;
            }
            if ( v31 )
              v15 = v31;
          }
          goto LABEL_21;
        }
LABEL_49:
        ++*(_DWORD *)(a1 + 16);
        BUG();
      }
LABEL_5:
      sub_22A9DC0(a1, 2 * v3);
      v11 = *(_DWORD *)(a1 + 24);
      if ( v11 )
      {
        v12 = v11 - 1;
        v13 = *(_QWORD *)(a1 + 8);
        v14 = v12 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v15 = (__int64 *)(v13 + 24LL * v14);
        v16 = *v15;
        v17 = *(_DWORD *)(a1 + 16) + 1;
        if ( *v15 != a2 )
        {
          v18 = 1;
          v19 = 0;
          while ( v16 != -4096 )
          {
            if ( v16 == -8192 && !v19 )
              v19 = v15;
            v14 = v12 & (v18 + v14);
            v15 = (__int64 *)(v13 + 24LL * v14);
            v16 = *v15;
            if ( *v15 == a2 )
              goto LABEL_21;
            ++v18;
          }
          if ( v19 )
            v15 = v19;
        }
        goto LABEL_21;
      }
      goto LABEL_49;
    }
    v25 = i + 1;
    v21 = v5 & ((_DWORD)v21 + i);
    v26 = (__int64 *)(v4 + 24 * v21);
    v20 = *v26;
    if ( *v26 == a2 )
      break;
  }
  if ( v26 == (__int64 *)(v4 + 24LL * v3) )
    goto LABEL_15;
  v8 = (__int64 *)(v4 + 24 * v21);
  return v8 + 1;
}
