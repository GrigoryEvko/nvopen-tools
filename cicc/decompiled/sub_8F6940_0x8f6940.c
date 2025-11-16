// Function: sub_8F6940
// Address: 0x8f6940
//
__int64 __fastcall sub_8F6940(_BYTE *a1, unsigned __int8 *a2, int a3)
{
  unsigned int v3; // r12d
  unsigned int v5; // eax
  int v6; // r11d
  int v7; // r15d
  char *v8; // rcx
  char *v9; // rsi
  int v10; // eax
  unsigned __int64 v11; // r13
  int v12; // edx
  unsigned __int64 v13; // rsi
  char *v14; // rcx
  int v15; // r15d
  int v16; // edx
  unsigned __int64 v17; // rsi
  __int64 *v18; // rsi
  __int128 *v19; // r15
  int v20; // r13d
  int v21; // eax
  __m128i v22; // xmm3
  __m128i v23; // xmm4
  __int128 *v24; // r15
  __int32 v25; // eax
  __m128i v26; // xmm1
  __m128i v27; // xmm2
  __int128 *v28; // rax
  int v29; // r13d
  char *v30; // [rsp+30h] [rbp-E8h]
  char *v31; // [rsp+30h] [rbp-E8h]
  int v32; // [rsp+38h] [rbp-E0h]
  int v33; // [rsp+3Ch] [rbp-DCh]
  int v34; // [rsp+40h] [rbp-D8h]
  __int128 *v35; // [rsp+40h] [rbp-D8h]
  int v36; // [rsp+40h] [rbp-D8h]
  __m128i v37; // [rsp+48h] [rbp-D0h] BYREF
  __m128i v38; // [rsp+58h] [rbp-C0h] BYREF
  __m128i v39; // [rsp+68h] [rbp-B0h] BYREF
  __m128i v40; // [rsp+78h] [rbp-A0h] BYREF
  __int64 v41; // [rsp+88h] [rbp-90h] BYREF
  int v42; // [rsp+90h] [rbp-88h]
  __int64 v43; // [rsp+94h] [rbp-84h]
  int v44; // [rsp+9Ch] [rbp-7Ch]
  __int16 v45; // [rsp+A0h] [rbp-78h]
  char v46; // [rsp+A2h] [rbp-76h]
  int v47; // [rsp+A4h] [rbp-74h]
  unsigned int v48; // [rsp+A8h] [rbp-70h] BYREF
  __int32 v49; // [rsp+ACh] [rbp-6Ch]
  int v50; // [rsp+B0h] [rbp-68h]
  char *v51; // [rsp+B8h] [rbp-60h]
  char *v52; // [rsp+C0h] [rbp-58h]
  char *v53; // [rsp+C8h] [rbp-50h]
  char *v54; // [rsp+D0h] [rbp-48h]
  int v55; // [rsp+D8h] [rbp-40h]

  if ( a3 <= 0 || a2 == 0 || !a1 )
    return (unsigned int)-3;
  sub_8F1450(&v48, a2, (char *)&a2[a3]);
  v5 = v48;
  if ( v48 == 2 )
  {
    v6 = 16;
    v38.m128i_i32[3] = 53;
    v37.m128i_i32[0] = 6;
    v33 = 0;
    if ( v55 <= 15 )
      v6 = v55;
    v7 = v6;
    v34 = v50 - v6;
    if ( v50 - v6 < 0 )
    {
      v33 = 1;
      v34 = v6 - v50;
    }
    if ( v50 > 309 )
    {
      v48 = 5;
      v5 = 5;
      goto LABEL_8;
    }
    if ( v50 < -324 )
    {
      v48 = 7;
      v5 = 7;
LABEL_7:
      if ( v5 - 1 <= 1 )
        return 0;
LABEL_8:
      v47 = 53;
      HIDWORD(v41) = v49;
      switch ( v5 )
      {
        case 3u:
          LODWORD(v41) = 3;
          v3 = 1;
          goto LABEL_11;
        case 4u:
          LODWORD(v41) = 4;
          v3 = 2 - ((v49 == 0) - 1);
          goto LABEL_11;
        case 5u:
          goto LABEL_25;
        case 6u:
          LODWORD(v41) = 6;
          v3 = 0;
          goto LABEL_11;
        case 7u:
          goto LABEL_16;
        default:
          sub_721090();
      }
    }
    if ( v55 > 15 )
      goto LABEL_46;
    if ( v34 <= 22 )
    {
      v48 = 2;
      v32 = 0;
      goto LABEL_37;
    }
    if ( v33 || v34 > 37LL - v55 )
    {
LABEL_46:
      v48 = 1;
      v32 = v34 & 0xF;
      v34 -= v32;
    }
    else
    {
      v48 = 2;
      v32 = 15 - v55;
      v34 -= 15 - v55;
    }
LABEL_37:
    v8 = v51;
    v9 = v52;
    if ( v51 == v52 )
    {
      v14 = v53;
      if ( v54 == v53 || !v6 )
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
          v30 = v8;
          sub_8EF960(&v37, dword_4F61F60);
          v8 = v30;
          if ( v11 )
          {
            v13 = v11;
            v11 = 0;
            sub_8F0790((__int64)&v37, v13);
            v9 = v52;
            v8 = v30;
          }
          else
          {
            v9 = v52;
          }
          v10 = 0;
        }
      }
      v14 = v53;
      if ( v54 == v53 || !v7 )
      {
LABEL_59:
        sub_8EF960(&v37, &qword_4F61E40[4 * v10]);
        if ( v11 )
          sub_8F0790((__int64)&v37, v11);
        if ( v37.m128i_i32[0] != 6 )
        {
          v18 = &qword_4F61E40[4 * v32];
          if ( v33 )
          {
            sub_8F0F10(&v37, (__int64)v18);
            if ( v34 > 22 )
            {
              v19 = xmmword_4F61DA0;
              v20 = 0;
              v21 = v34 >> 4;
              while ( 1 )
              {
                if ( (v21 & 1) != 0 )
                {
                  v36 = v21;
                  sub_8F0F10(&v37, (__int64)v19);
                  v21 = v36;
                }
                ++v20;
                v19 += 2;
                v21 >>= 1;
                if ( !v21 )
                  goto LABEL_17;
                if ( v20 > 3 )
                {
                  v22 = _mm_loadu_si128(&v37);
                  v23 = _mm_loadu_si128(&v38);
                  v45 = 0;
                  v24 = &xmmword_4F61DA0[2 * v20];
                  v43 = 0x10000000000000LL;
                  v44 = 0;
                  v41 = 2;
                  v42 = -1021;
                  v46 = 0;
                  v47 = 53;
                  v39 = v22;
                  v40 = v23;
                  sub_8F0F10(&v37, (__int64)v24);
                  v33 = 0;
                  while ( sub_8F0EB0(&v37, &v41) )
                  {
                    if ( v37.m128i_i32[0] == 6 )
                      v37.m128i_i32[2] = -1073;
                    v25 = -1021 - v37.m128i_i32[2];
                    v26 = _mm_loadu_si128(&v40);
                    v37 = _mm_loadu_si128(&v39);
                    v37.m128i_i32[2] += v25;
                    v27 = _mm_loadu_si128(&v37);
                    v33 += v25;
                    v38 = v26;
                    v39 = v27;
                    sub_8F0F10(&v37, (__int64)v24);
                  }
                  goto LABEL_18;
                }
              }
            }
            sub_8F0F10(&v37, (__int64)&qword_4F61E40[4 * v34]);
            v33 = 0;
          }
          else
          {
            sub_8EF960(&v37, v18);
            v28 = xmmword_4F61DA0;
            v29 = v34 >> 4;
            if ( v34 > 22 )
            {
              do
              {
                if ( (v29 & 1) != 0 )
                {
                  v35 = v28;
                  sub_8EF960(&v37, v28);
                  v28 = v35;
                }
                v28 += 2;
                v29 >>= 1;
              }
              while ( v29 );
LABEL_17:
              v33 = 0;
              goto LABEL_18;
            }
            sub_8EF960(&v37, &qword_4F61E40[4 * v34]);
          }
LABEL_18:
          v5 = v48;
          if ( v49 )
            v37.m128i_i32[1] = v37.m128i_i32[1] == 0;
          if ( v48 != 1 )
            goto LABEL_75;
          if ( (unsigned int)(v37.m128i_i32[0] - 6) <= 1 )
          {
            v37.m128i_i64[0] = 2;
            v37.m128i_i64[1] = 4294966223LL;
            v38.m128i_i16[0] = 0;
            v38.m128i_i8[2] = 16;
            v38.m128i_i32[3] = 53;
          }
          else if ( v37.m128i_i32[0] != 2 )
          {
LABEL_23:
            if ( v37.m128i_i32[2] > 1024 )
            {
              v48 = 5;
              v47 = 53;
              HIDWORD(v41) = v49;
LABEL_25:
              LODWORD(v41) = 4;
              v3 = 4;
              goto LABEL_11;
            }
            if ( v37.m128i_i32[2] < -1073 )
            {
              v48 = 7;
              v47 = 53;
              HIDWORD(v41) = v49;
LABEL_16:
              LODWORD(v41) = 6;
              v3 = 5;
LABEL_11:
              sub_8EFFA0(a1, (int *)&v41);
              return v3;
            }
            v37.m128i_i32[1] = v49;
            v5 = v48;
LABEL_75:
            if ( v5 - 1 <= 1 )
            {
              sub_8EFFA0(a1, v37.m128i_i32);
              v5 = v48;
            }
            goto LABEL_6;
          }
          sub_8F1B70((__int64)&v37, (__int64)&v48, v33);
          goto LABEL_23;
        }
LABEL_57:
        v48 = 7;
        if ( v49 )
        {
          v5 = 7;
          goto LABEL_8;
        }
        v47 = 53;
        HIDWORD(v41) = 0;
        goto LABEL_16;
      }
    }
    v15 = (_DWORD)v14 + v7;
    do
    {
      if ( v10 == 9 )
      {
        v31 = v14;
        sub_8EF960(&v37, dword_4F61F60);
        v10 = 1;
        v14 = v31;
        if ( v11 )
        {
          v17 = v11;
          v11 = 0;
          sub_8F0790((__int64)&v37, v17);
          v10 = 1;
          v14 = v31;
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
    while ( v15 != (_DWORD)v14 && v54 != v14 );
    goto LABEL_59;
  }
LABEL_6:
  if ( v5 )
    goto LABEL_7;
  return (unsigned int)-3;
}
