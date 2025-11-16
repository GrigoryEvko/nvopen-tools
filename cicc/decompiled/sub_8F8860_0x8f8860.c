// Function: sub_8F8860
// Address: 0x8f8860
//
__int64 __fastcall sub_8F8860(_BYTE *a1, unsigned __int8 *a2, int a3)
{
  unsigned int v3; // r12d
  unsigned int v5; // eax
  int v6; // r11d
  int v7; // r15d
  char *v8; // rcx
  char *v9; // rsi
  int v10; // eax
  unsigned __int64 v11; // rbx
  int v12; // edx
  unsigned __int64 v13; // rsi
  char *v14; // rcx
  int v15; // r15d
  int v16; // edx
  unsigned __int64 v17; // rsi
  __int64 *v18; // rsi
  __int128 *v19; // r15
  int v20; // eax
  int v21; // ebx
  __int128 *v22; // rbx
  int v23; // eax
  __m128i v24; // xmm3
  __m128i v25; // xmm4
  __int128 *v26; // rbx
  __int32 v27; // eax
  __m128i v28; // xmm1
  __m128i v29; // xmm2
  int v30; // [rsp+Ch] [rbp-E4h]
  char *v31; // [rsp+10h] [rbp-E0h]
  char *v32; // [rsp+10h] [rbp-E0h]
  int v33; // [rsp+18h] [rbp-D8h]
  int v34; // [rsp+1Ch] [rbp-D4h]
  int v35; // [rsp+1Ch] [rbp-D4h]
  int v36; // [rsp+1Ch] [rbp-D4h]
  __m256i v37; // [rsp+20h] [rbp-D0h] BYREF
  __m128i v38; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v39; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v40; // [rsp+60h] [rbp-90h] BYREF
  int v41; // [rsp+68h] [rbp-88h]
  __int64 v42; // [rsp+6Ch] [rbp-84h]
  int v43; // [rsp+74h] [rbp-7Ch]
  __int16 v44; // [rsp+78h] [rbp-78h]
  char v45; // [rsp+7Ah] [rbp-76h]
  int v46; // [rsp+7Ch] [rbp-74h]
  unsigned int v47; // [rsp+80h] [rbp-70h] BYREF
  __int32 v48; // [rsp+84h] [rbp-6Ch]
  int v49; // [rsp+88h] [rbp-68h]
  char *v50; // [rsp+90h] [rbp-60h]
  char *v51; // [rsp+98h] [rbp-58h]
  char *v52; // [rsp+A0h] [rbp-50h]
  char *v53; // [rsp+A8h] [rbp-48h]
  int v54; // [rsp+B0h] [rbp-40h]

  if ( a3 <= 0 || a2 == 0 || !a1 )
    return (unsigned int)-3;
  sub_8F1450(&v47, a2, (char *)&a2[a3]);
  v5 = v47;
  if ( v47 == 2 )
  {
    v6 = 34;
    v37.m256i_i32[7] = 113;
    v37.m256i_i32[0] = 6;
    v33 = 0;
    if ( v54 <= 33 )
      v6 = v54;
    v7 = v6;
    v34 = v49 - v6;
    if ( v49 - v6 < 0 )
    {
      v33 = 1;
      v34 = v6 - v49;
    }
    if ( v49 > 4933 )
    {
      v47 = 5;
      v5 = 5;
      goto LABEL_8;
    }
    if ( v49 < -4966 )
    {
      v47 = 7;
      v5 = 7;
LABEL_7:
      if ( v5 - 1 <= 1 )
        return 0;
LABEL_8:
      v46 = 113;
      HIDWORD(v40) = v48;
      switch ( v5 )
      {
        case 3u:
          LODWORD(v40) = 3;
          v3 = 1;
          goto LABEL_11;
        case 4u:
          LODWORD(v40) = 4;
          v3 = 2 - ((v48 == 0) - 1);
          goto LABEL_11;
        case 5u:
          goto LABEL_25;
        case 6u:
          LODWORD(v40) = 6;
          v3 = 0;
          goto LABEL_11;
        case 7u:
          goto LABEL_16;
        default:
          sub_721090();
      }
    }
    if ( v54 > 33 )
      goto LABEL_46;
    if ( v34 <= 48 )
    {
      v47 = 2;
      v30 = 0;
      goto LABEL_37;
    }
    if ( v33 || v34 > 81LL - v54 )
    {
LABEL_46:
      v47 = 1;
      v30 = v34 & 0xF;
      v34 -= v30;
    }
    else
    {
      v47 = 2;
      v30 = 33 - v54;
      v34 -= 33 - v54;
    }
LABEL_37:
    v8 = v50;
    v9 = v51;
    if ( v50 == v51 )
    {
      v14 = v52;
      if ( v53 == v52 || !v6 )
        goto LABEL_57;
      v11 = 0;
      v10 = 0;
    }
    else
    {
      if ( !v6 )
        goto LABEL_57;
      v10 = 0;
      v11 = 0;
      while ( 1 )
      {
        v12 = *v8++;
        --v7;
        ++v10;
        v11 = v12 - 48 + 10 * v11;
        if ( v8 == v9 )
          break;
        if ( !v7 )
          goto LABEL_59;
        if ( v10 == 9 )
        {
          v31 = v8;
          sub_8EF960(&v37, dword_4F60820);
          v8 = v31;
          if ( v11 )
          {
            v13 = v11;
            v11 = 0;
            sub_8F0790((__int64)&v37, v13);
            v9 = v51;
            v8 = v31;
          }
          else
          {
            v9 = v51;
          }
          v10 = 0;
        }
      }
      v14 = v52;
      if ( v53 == v52 || !v7 )
      {
LABEL_59:
        sub_8EF960(&v37, &qword_4F60700[4 * v10]);
        if ( v11 )
          sub_8F0790((__int64)&v37, v11);
        if ( v37.m256i_i32[0] != 6 )
        {
          v18 = &qword_4F60700[4 * v30];
          if ( v33 )
          {
            sub_8F0F10(&v37, (__int64)v18);
            if ( v34 > 48 )
            {
              v19 = xmmword_4F605E0;
              v20 = 0;
              v21 = v34 >> 4;
              while ( 1 )
              {
                if ( (v21 & 1) != 0 )
                {
                  v35 = v20;
                  sub_8F0F10(&v37, (__int64)v19);
                  v20 = v35;
                }
                ++v20;
                v19 += 2;
                v21 >>= 1;
                if ( !v21 )
                  goto LABEL_17;
                if ( v20 > 7 )
                {
                  v45 = 1;
                  v24 = _mm_loadu_si128((const __m128i *)&v37);
                  v44 = 0;
                  v25 = _mm_loadu_si128((const __m128i *)&v37.m256i_u64[2]);
                  v26 = &xmmword_4F605E0[2 * v20];
                  v40 = 2;
                  v43 = 0;
                  v41 = -16381;
                  v42 = 0;
                  v46 = 113;
                  v38 = v24;
                  v39 = v25;
                  sub_8F0F10(&v37, (__int64)v26);
                  v33 = 0;
                  while ( sub_8F0EB0(&v37, &v40) )
                  {
                    if ( v37.m256i_i32[0] == 6 )
                      v37.m256i_i32[2] = -16493;
                    v27 = -16381 - v37.m256i_i32[2];
                    v28 = _mm_loadu_si128(&v39);
                    *(__m128i *)v37.m256i_i8 = _mm_loadu_si128(&v38);
                    v37.m256i_i32[2] += v27;
                    v29 = _mm_loadu_si128((const __m128i *)&v37);
                    v33 += v27;
                    *(__m128i *)&v37.m256i_u64[2] = v28;
                    v38 = v29;
                    sub_8F0F10(&v37, (__int64)v26);
                  }
                  goto LABEL_18;
                }
              }
            }
            sub_8F0F10(&v37, (__int64)&qword_4F60700[4 * v34]);
            v33 = 0;
          }
          else
          {
            sub_8EF960(&v37, v18);
            v22 = xmmword_4F605E0;
            v23 = v34 >> 4;
            if ( v34 > 48 )
            {
              do
              {
                if ( (v23 & 1) != 0 )
                {
                  v36 = v23;
                  sub_8EF960(&v37, v22);
                  v23 = v36;
                }
                v22 += 2;
                v23 >>= 1;
              }
              while ( v23 );
LABEL_17:
              v33 = 0;
              goto LABEL_18;
            }
            sub_8EF960(&v37, &qword_4F60700[4 * v34]);
          }
LABEL_18:
          v5 = v47;
          if ( v48 )
            v37.m256i_i32[1] = v37.m256i_i32[1] == 0;
          if ( v47 != 1 )
            goto LABEL_70;
          if ( (unsigned int)(v37.m256i_i32[0] - 6) <= 1 )
          {
            v37.m256i_i64[0] = 2;
            v37.m256i_i32[2] = -16493;
            *(__int64 *)((char *)&v37.m256i_i64[1] + 4) = 0;
            v37.m256i_i32[5] = 0;
            v37.m256i_i16[12] = 0;
            v37.m256i_i8[26] = 1;
            v37.m256i_i32[7] = 113;
          }
          else if ( v37.m256i_i32[0] != 2 )
          {
LABEL_23:
            if ( v37.m256i_i32[2] > 0x4000 )
            {
              v47 = 5;
              v46 = 113;
              HIDWORD(v40) = v48;
LABEL_25:
              LODWORD(v40) = 4;
              v3 = 4;
              goto LABEL_11;
            }
            if ( v37.m256i_i32[2] < -16493 )
            {
              v47 = 7;
              v46 = 113;
              HIDWORD(v40) = v48;
LABEL_16:
              LODWORD(v40) = 6;
              v3 = 5;
LABEL_11:
              sub_8F0450(a1, (int *)&v40);
              return v3;
            }
            v37.m256i_i32[1] = v48;
            v5 = v47;
LABEL_70:
            if ( v5 - 1 <= 1 )
            {
              sub_8F0450(a1, v37.m256i_i32);
              v5 = v47;
            }
            goto LABEL_6;
          }
          sub_8F1B70((__int64)&v37, (__int64)&v47, v33);
          goto LABEL_23;
        }
LABEL_57:
        v47 = 7;
        if ( v48 )
        {
          v5 = 7;
          goto LABEL_8;
        }
        v46 = 113;
        HIDWORD(v40) = 0;
        goto LABEL_16;
      }
    }
    v15 = (_DWORD)v14 + v7;
    do
    {
      if ( v10 == 9 )
      {
        v32 = v14;
        sub_8EF960(&v37, dword_4F60820);
        v10 = 1;
        v14 = v32;
        if ( v11 )
        {
          v17 = v11;
          v11 = 0;
          sub_8F0790((__int64)&v37, v17);
          v10 = 1;
          v14 = v32;
        }
      }
      else
      {
        ++v10;
        v11 *= 10LL;
      }
      v16 = *v14++;
      v11 += v16 - 48;
    }
    while ( v15 != (_DWORD)v14 && v53 != v14 );
    goto LABEL_59;
  }
LABEL_6:
  if ( v5 )
    goto LABEL_7;
  return (unsigned int)-3;
}
