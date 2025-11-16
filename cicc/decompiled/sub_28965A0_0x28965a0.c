// Function: sub_28965A0
// Address: 0x28965a0
//
__int64 __fastcall sub_28965A0(unsigned __int8 *a1, __int64 a2)
{
  int v2; // ecx
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rax
  unsigned __int8 *v6; // rdx
  __int64 v7; // r10
  __int64 v8; // r9
  int v9; // r11d
  __int64 v10; // rcx
  unsigned int v11; // eax
  __int64 *v12; // rsi
  __int64 v13; // r8
  int v14; // esi
  int v15; // r12d
  unsigned __int8 *v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rdi
  unsigned int v20; // edx
  __int64 v21; // r8
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r10
  __int64 v25; // rdx
  __int64 v27; // rdx
  __int64 v28; // r8
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // r11
  __int64 v32; // r10
  __int64 v33; // rdx
  __int64 v34; // r10
  __int64 v35; // r9
  int v36; // esi
  int v37; // r10d
  __int64 v38; // [rsp+14h] [rbp-3Ch] BYREF
  __int32 v39; // [rsp+1Ch] [rbp-34h]
  __m128i v40; // [rsp+20h] [rbp-30h] BYREF
  __int128 v41; // [rsp+30h] [rbp-20h]

  v2 = *a1;
  if ( (_BYTE)v2 == 85 )
  {
    v3 = *((_QWORD *)a1 - 4);
    if ( !v3 )
      goto LABEL_10;
    if ( !*(_BYTE *)v3
      && (v4 = *(_QWORD *)(v3 + 24), v4 == *((_QWORD *)a1 + 10))
      && *(_DWORD *)(v3 + 36) == 233
      && (v23 = *((_DWORD *)a1 + 1) & 0x7FFFFFF, (v24 = *(_QWORD *)&a1[32 * (2 - v23)]) != 0)
      && *(_QWORD *)&a1[32 * (3 - v23)] )
    {
      v25 = *(_QWORD *)&a1[32 * (4 - v23)];
      if ( v25 )
      {
        sub_28940A0((__int64)&v38, v24, v25);
        v40.m128i_i8[12] = 1;
        v40.m128i_i64[0] = v38;
        v40.m128i_i32[2] = v39;
        return _mm_loadu_si128(&v40).m128i_i64[0];
      }
    }
    else
    {
      if ( *(_BYTE *)v3 )
        goto LABEL_8;
      v4 = *(_QWORD *)(v3 + 24);
    }
    if ( v4 == *((_QWORD *)a1 + 10) && *(_DWORD *)(v3 + 36) == 234 )
    {
      v30 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
      v31 = *(_QWORD *)&a1[32 * (1 - v30)];
      if ( v31 )
      {
        v32 = *(_QWORD *)&a1[32 * (2 - v30)];
        if ( v32 )
        {
          sub_28940A0((__int64)&v38, v32, v31);
          v40.m128i_i8[12] = 1;
          v40.m128i_i64[0] = v38;
          v40.m128i_i32[2] = v39;
          return _mm_loadu_si128(&v40).m128i_i64[0];
        }
        goto LABEL_49;
      }
    }
LABEL_8:
    if ( *(_BYTE *)v3 )
      goto LABEL_38;
    v4 = *(_QWORD *)(v3 + 24);
LABEL_49:
    if ( v4 == *((_QWORD *)a1 + 10) && *(_DWORD *)(v3 + 36) == 232 )
    {
      v33 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
      v34 = *(_QWORD *)&a1[32 * (4 - v33)];
      if ( v34 )
      {
        v35 = *(_QWORD *)&a1[32 * (5 - v33)];
        if ( v35 )
        {
          sub_28940A0((__int64)&v38, v35, v34);
          v40.m128i_i8[12] = 1;
          v40.m128i_i64[0] = v38;
          v40.m128i_i32[2] = v39;
          return _mm_loadu_si128(&v40).m128i_i64[0];
        }
      }
    }
LABEL_38:
    if ( !*(_BYTE *)v3 && *(_QWORD *)(v3 + 24) == *((_QWORD *)a1 + 10) && *(_DWORD *)(v3 + 36) == 231 )
    {
      v27 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
      v28 = *(_QWORD *)&a1[32 * (3 - v27)];
      if ( v28 )
      {
        v29 = *(_QWORD *)&a1[32 * (4 - v27)];
        if ( v29 )
        {
          sub_28940A0((__int64)&v38, v28, v29);
          v40.m128i_i8[12] = 1;
          v40.m128i_i64[0] = v38;
          v40.m128i_i32[2] = v39;
          return _mm_loadu_si128(&v40).m128i_i64[0];
        }
      }
    }
LABEL_10:
    if ( (unsigned int)(v2 - 41) <= 6 )
      goto LABEL_11;
LABEL_37:
    BYTE12(v41) = 0;
    return v41;
  }
  if ( (_BYTE)v2 != 62 )
  {
    if ( (unsigned __int8)v2 <= 0x1Cu )
    {
LABEL_11:
      v5 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
      if ( (a1[7] & 0x40) != 0 )
      {
        v6 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
        a1 = &v6[v5];
      }
      else
      {
        v6 = &a1[-v5];
      }
      if ( a1 != v6 )
      {
        v7 = *(_QWORD *)(a2 + 8);
        v8 = *(unsigned int *)(a2 + 24);
        v9 = v8 - 1;
        do
        {
          v10 = *(_QWORD *)v6;
          if ( (_DWORD)v8 )
          {
            v11 = v9 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
            v12 = (__int64 *)(v7 + 24LL * v11);
            v13 = *v12;
            if ( v10 == *v12 )
            {
LABEL_15:
              if ( (__int64 *)(v7 + 24 * v8) != v12 )
                goto LABEL_30;
            }
            else
            {
              v14 = 1;
              while ( v13 != -4096 )
              {
                v15 = v14 + 1;
                v11 = v9 & (v14 + v11);
                v12 = (__int64 *)(v7 + 24LL * v11);
                v13 = *v12;
                if ( v10 == *v12 )
                  goto LABEL_15;
                v14 = v15;
              }
            }
          }
          v6 += 32;
        }
        while ( a1 != v6 );
      }
      goto LABEL_37;
    }
    goto LABEL_10;
  }
  if ( (a1[7] & 0x40) != 0 )
    v16 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
  else
    v16 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  v17 = *(_QWORD *)v16;
  if ( !*(_QWORD *)v16 )
    goto LABEL_37;
  v18 = *(unsigned int *)(a2 + 24);
  if ( !(_DWORD)v18 )
    goto LABEL_37;
  v19 = *(_QWORD *)(a2 + 8);
  v20 = (v18 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
  v12 = (__int64 *)(v19 + 24LL * v20);
  v21 = *v12;
  if ( v17 != *v12 )
  {
    v36 = 1;
    while ( v21 != -4096 )
    {
      v37 = v36 + 1;
      v20 = (v18 - 1) & (v36 + v20);
      v12 = (__int64 *)(v19 + 24LL * v20);
      v21 = *v12;
      if ( v17 == *v12 )
        goto LABEL_29;
      v36 = v37;
    }
    goto LABEL_37;
  }
LABEL_29:
  if ( v12 == (__int64 *)(v19 + 24 * v18) )
    goto LABEL_37;
LABEL_30:
  v22 = v12[1];
  v40.m128i_i8[12] = 1;
  v40.m128i_i64[0] = v22;
  v40.m128i_i32[2] = *((_DWORD *)v12 + 4);
  return _mm_loadu_si128(&v40).m128i_i64[0];
}
