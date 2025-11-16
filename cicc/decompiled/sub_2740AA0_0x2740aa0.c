// Function: sub_2740AA0
// Address: 0x2740aa0
//
__int64 __fastcall sub_2740AA0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r9
  __int64 result; // rax
  __int64 v5; // r11
  __int64 v6; // rdx
  int v7; // r13d
  __int64 *v8; // r12
  unsigned int i; // esi
  unsigned int v11; // r8d
  __int64 v12; // rdi
  __int64 v13; // r14
  unsigned int v14; // r13d
  int v15; // esi
  int v16; // esi
  __int64 v17; // r8
  unsigned int v18; // edx
  __int64 v19; // rcx
  __int64 v20; // rdi
  int v21; // r15d
  __int64 v22; // r10
  int v23; // edx
  int v24; // edx
  __int64 v25; // r8
  unsigned int v26; // r15d
  __int64 v27; // rdi
  int v28; // r10d
  __int64 v29; // rsi
  __int64 *v30; // [rsp-48h] [rbp-48h]
  __int64 *v31; // [rsp-48h] [rbp-48h]
  __int64 v32; // [rsp-40h] [rbp-40h]
  int v33; // [rsp-40h] [rbp-40h]
  __int64 v34; // [rsp-40h] [rbp-40h]

  v3 = &a2[a3];
  *(_QWORD *)(a1 + 600) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  result = 0x400000000LL;
  *(_QWORD *)(a1 + 16) = 0x400000000LL;
  *(_QWORD *)(a1 + 608) = 0;
  *(_QWORD *)(a1 + 616) = 0;
  *(_DWORD *)(a1 + 624) = 0;
  *(_QWORD *)a1 = a3;
  if ( a2 != v3 )
  {
    v5 = a1 + 600;
    v6 = 0;
    v7 = 0;
    v8 = a2;
    for ( i = 0; ; i = *(_DWORD *)(a1 + 624) )
    {
      v13 = *v8;
      v14 = v7 + 1;
      if ( i )
      {
        v11 = (i - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        result = v6 + 16LL * v11;
        v12 = *(_QWORD *)result;
        if ( v13 == *(_QWORD *)result )
          goto LABEL_4;
        v33 = 1;
        v19 = 0;
        while ( v12 != -4096 )
        {
          if ( v19 || v12 != -8192 )
            result = v19;
          v11 = (i - 1) & (v33 + v11);
          v12 = *(_QWORD *)(v6 + 16LL * v11);
          if ( v13 == v12 )
            goto LABEL_4;
          ++v33;
          v19 = result;
          result = v6 + 16LL * v11;
        }
        if ( !v19 )
          v19 = result;
        ++*(_QWORD *)(a1 + 600);
        if ( 4 * v14 < 3 * i )
        {
          result = v14;
          if ( i - *(_DWORD *)(a1 + 620) - v14 <= i >> 3 )
          {
            v31 = v3;
            v34 = v5;
            sub_D39D40(v5, i);
            v23 = *(_DWORD *)(a1 + 624);
            if ( !v23 )
            {
LABEL_47:
              ++*(_DWORD *)(a1 + 616);
              BUG();
            }
            v24 = v23 - 1;
            v25 = 0;
            v5 = v34;
            v26 = v24 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v3 = v31;
            v27 = *(_QWORD *)(a1 + 608);
            v28 = 1;
            result = (unsigned int)(*(_DWORD *)(a1 + 616) + 1);
            v19 = v27 + 16LL * v26;
            v29 = *(_QWORD *)v19;
            if ( v13 != *(_QWORD *)v19 )
            {
              while ( v29 != -4096 )
              {
                if ( !v25 && v29 == -8192 )
                  v25 = v19;
                v26 = v24 & (v28 + v26);
                v19 = v27 + 16LL * v26;
                v29 = *(_QWORD *)v19;
                if ( v13 == *(_QWORD *)v19 )
                  goto LABEL_22;
                ++v28;
              }
              if ( v25 )
                v19 = v25;
            }
          }
          goto LABEL_22;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 600);
      }
      v30 = v3;
      v32 = v5;
      sub_D39D40(v5, 2 * i);
      v15 = *(_DWORD *)(a1 + 624);
      if ( !v15 )
        goto LABEL_47;
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 608);
      v5 = v32;
      v3 = v30;
      v18 = v16 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      result = (unsigned int)(*(_DWORD *)(a1 + 616) + 1);
      v19 = v17 + 16LL * v18;
      v20 = *(_QWORD *)v19;
      if ( v13 != *(_QWORD *)v19 )
      {
        v21 = 1;
        v22 = 0;
        while ( v20 != -4096 )
        {
          if ( v20 == -8192 && !v22 )
            v22 = v19;
          v18 = v16 & (v21 + v18);
          v19 = v17 + 16LL * v18;
          v20 = *(_QWORD *)v19;
          if ( v13 == *(_QWORD *)v19 )
            goto LABEL_22;
          ++v21;
        }
        if ( v22 )
          v19 = v22;
      }
LABEL_22:
      *(_DWORD *)(a1 + 616) = result;
      if ( *(_QWORD *)v19 != -4096 )
        --*(_DWORD *)(a1 + 620);
      *(_QWORD *)v19 = v13;
      *(_DWORD *)(v19 + 8) = v14;
LABEL_4:
      if ( v3 == ++v8 )
        return result;
      v7 = *(_DWORD *)(a1 + 616);
      v6 = *(_QWORD *)(a1 + 608);
    }
  }
  return result;
}
