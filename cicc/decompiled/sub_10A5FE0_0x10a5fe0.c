// Function: sub_10A5FE0
// Address: 0x10a5fe0
//
__int64 __fastcall sub_10A5FE0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  __int64 v5; // r15
  int v6; // r11d
  __int64 v7; // r9
  __int64 v8; // rdx
  unsigned int v9; // edi
  __int64 v10; // rcx
  unsigned int v11; // esi
  __int64 v12; // r13
  __int64 v13; // r8
  int v14; // eax
  int v15; // edi
  __int64 v16; // rsi
  unsigned int v17; // eax
  int v18; // ecx
  int v19; // r11d
  __int64 v20; // r10
  int v21; // eax
  int v22; // eax
  int v23; // eax
  unsigned int v24; // r14d
  __int64 v25; // rdi
  int v26; // r10d
  __int64 v27; // rsi
  unsigned int v28; // [rsp+0h] [rbp-40h]
  unsigned int v29; // [rsp+0h] [rbp-40h]
  const void *v30; // [rsp+8h] [rbp-38h]

  result = a1 + 16;
  v3 = *(_QWORD *)(a2 + 16);
  v30 = (const void *)(a1 + 16);
  if ( v3 )
  {
    v5 = a1 + 2064;
    while ( 1 )
    {
      v11 = *(_DWORD *)(a1 + 2088);
      v12 = *(_QWORD *)(v3 + 24);
      v13 = *(unsigned int *)(a1 + 8);
      if ( !v11 )
        break;
      v6 = 1;
      v7 = *(_QWORD *)(a1 + 2072);
      v8 = 0;
      v9 = (v11 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      result = v7 + 16LL * v9;
      v10 = *(_QWORD *)result;
      if ( v12 == *(_QWORD *)result )
      {
LABEL_4:
        v3 = *(_QWORD *)(v3 + 8);
        if ( !v3 )
          return result;
      }
      else
      {
        while ( v10 != -4096 )
        {
          if ( v10 != -8192 || v8 )
            result = v8;
          v9 = (v11 - 1) & (v6 + v9);
          v10 = *(_QWORD *)(v7 + 16LL * v9);
          if ( v12 == v10 )
            goto LABEL_4;
          ++v6;
          v8 = result;
          result = v7 + 16LL * v9;
        }
        if ( !v8 )
          v8 = result;
        v21 = *(_DWORD *)(a1 + 2080);
        ++*(_QWORD *)(a1 + 2064);
        v18 = v21 + 1;
        if ( 4 * (v21 + 1) < 3 * v11 )
        {
          if ( v11 - *(_DWORD *)(a1 + 2084) - v18 <= v11 >> 3 )
          {
            v29 = v13;
            sub_9BAAD0(v5, v11);
            v22 = *(_DWORD *)(a1 + 2088);
            if ( !v22 )
            {
LABEL_46:
              ++*(_DWORD *)(a1 + 2080);
              BUG();
            }
            v23 = v22 - 1;
            v7 = 0;
            v13 = v29;
            v24 = v23 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
            v25 = *(_QWORD *)(a1 + 2072);
            v26 = 1;
            v18 = *(_DWORD *)(a1 + 2080) + 1;
            v8 = v25 + 16LL * v24;
            v27 = *(_QWORD *)v8;
            if ( v12 != *(_QWORD *)v8 )
            {
              while ( v27 != -4096 )
              {
                if ( !v7 && v27 == -8192 )
                  v7 = v8;
                v24 = v23 & (v26 + v24);
                v8 = v25 + 16LL * v24;
                v27 = *(_QWORD *)v8;
                if ( v12 == *(_QWORD *)v8 )
                  goto LABEL_23;
                ++v26;
              }
              if ( v7 )
                v8 = v7;
            }
          }
          goto LABEL_23;
        }
LABEL_7:
        v28 = v13;
        sub_9BAAD0(v5, 2 * v11);
        v14 = *(_DWORD *)(a1 + 2088);
        if ( !v14 )
          goto LABEL_46;
        v15 = v14 - 1;
        v16 = *(_QWORD *)(a1 + 2072);
        v13 = v28;
        v17 = (v14 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v18 = *(_DWORD *)(a1 + 2080) + 1;
        v8 = v16 + 16LL * v17;
        v7 = *(_QWORD *)v8;
        if ( v12 != *(_QWORD *)v8 )
        {
          v19 = 1;
          v20 = 0;
          while ( v7 != -4096 )
          {
            if ( v7 == -8192 && !v20 )
              v20 = v8;
            v17 = v15 & (v19 + v17);
            v8 = v16 + 16LL * v17;
            v7 = *(_QWORD *)v8;
            if ( v12 == *(_QWORD *)v8 )
              goto LABEL_23;
            ++v19;
          }
          if ( v20 )
            v8 = v20;
        }
LABEL_23:
        *(_DWORD *)(a1 + 2080) = v18;
        if ( *(_QWORD *)v8 != -4096 )
          --*(_DWORD *)(a1 + 2084);
        *(_QWORD *)v8 = v12;
        *(_DWORD *)(v8 + 8) = v13;
        result = *(unsigned int *)(a1 + 8);
        if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          sub_C8D5F0(a1, v30, result + 1, 8u, v13, v7);
          result = *(unsigned int *)(a1 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a1 + 8 * result) = v12;
        ++*(_DWORD *)(a1 + 8);
        v3 = *(_QWORD *)(v3 + 8);
        if ( !v3 )
          return result;
      }
    }
    ++*(_QWORD *)(a1 + 2064);
    goto LABEL_7;
  }
  return result;
}
