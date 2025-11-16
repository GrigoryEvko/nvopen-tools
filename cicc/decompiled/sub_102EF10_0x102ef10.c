// Function: sub_102EF10
// Address: 0x102ef10
//
_QWORD *__fastcall sub_102EF10(__int64 a1, int a2)
{
  unsigned __int64 v2; // rcx
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rcx
  __int64 v9; // rbx
  _QWORD *i; // rcx
  __int64 v11; // r14
  __int64 j; // rax
  __int64 v13; // r15
  int v14; // eax
  int v15; // esi
  __int64 v16; // rdi
  unsigned int v17; // ecx
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // rdx
  _QWORD *k; // rdx
  int v22; // r10d
  __int64 v23; // r9
  __int64 v24; // rcx
  __int64 v25; // [rsp+0h] [rbp-80h]
  __int64 v26; // [rsp+0h] [rbp-80h]
  __int64 v27; // [rsp+8h] [rbp-78h]
  _QWORD v28[2]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v29; // [rsp+20h] [rbp-60h]
  _QWORD v30[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v31; // [rsp+40h] [rbp-40h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = (_QWORD *)sub_C7D670(48LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v27 = 48 * v4;
    v9 = v5 + 48 * v4;
    for ( i = &result[6 * v8]; i != result; result += 6 )
    {
      if ( result )
      {
        *result = 0;
        result[1] = 0;
        result[2] = -4096;
      }
    }
    v28[0] = 0;
    v28[1] = 0;
    v29 = -4096;
    v30[0] = 0;
    v30[1] = 0;
    v31 = -8192;
    if ( v9 != v5 )
    {
      v11 = v5;
      for ( j = -4096; ; j = v29 )
      {
        v13 = *(_QWORD *)(v11 + 16);
        if ( v13 != j )
        {
          j = v31;
          if ( v13 != v31 )
          {
            v14 = *(_DWORD *)(a1 + 24);
            if ( !v14 )
              BUG();
            v15 = v14 - 1;
            v16 = *(_QWORD *)(a1 + 8);
            v17 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v18 = v16 + 48LL * v17;
            v19 = *(_QWORD *)(v18 + 16);
            if ( v13 != v19 )
            {
              v22 = 1;
              v23 = 0;
              while ( v19 != -4096 )
              {
                if ( v19 == -8192 && !v23 )
                  v23 = v18;
                v17 = v15 & (v22 + v17);
                v18 = v16 + 48LL * v17;
                v19 = *(_QWORD *)(v18 + 16);
                if ( v13 == v19 )
                  goto LABEL_15;
                ++v22;
              }
              if ( v23 )
              {
                v24 = *(_QWORD *)(v23 + 16);
                v18 = v23;
              }
              else
              {
                v24 = *(_QWORD *)(v18 + 16);
              }
              if ( v13 != v24 )
              {
                if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
                {
                  v25 = v18;
                  sub_BD60C0((_QWORD *)v18);
                  v18 = v25;
                }
                *(_QWORD *)(v18 + 16) = v13;
                if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
                {
                  v26 = v18;
                  sub_BD73F0(v18);
                  v18 = v26;
                }
              }
            }
LABEL_15:
            *(__m128i *)(v18 + 24) = _mm_loadu_si128((const __m128i *)(v11 + 24));
            *(_QWORD *)(v18 + 40) = *(_QWORD *)(v11 + 40);
            ++*(_DWORD *)(a1 + 16);
            j = *(_QWORD *)(v11 + 16);
          }
        }
        if ( j != -4096 && j != 0 && j != -8192 )
          sub_BD60C0((_QWORD *)v11);
        v11 += 48;
        if ( v9 == v11 )
          break;
      }
      if ( v31 != 0 && v31 != -4096 && v31 != -8192 )
        sub_BD60C0(v30);
      if ( v29 != -8192 && v29 != -4096 && v29 != 0 )
        sub_BD60C0(v28);
    }
    return (_QWORD *)sub_C7D6A0(v5, v27, 8);
  }
  else
  {
    v20 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[6 * v20]; k != result; result += 6 )
    {
      if ( result )
      {
        *result = 0;
        result[1] = 0;
        result[2] = -4096;
      }
    }
  }
  return result;
}
