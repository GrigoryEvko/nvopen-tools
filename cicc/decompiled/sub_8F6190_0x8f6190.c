// Function: sub_8F6190
// Address: 0x8f6190
//
__int64 __fastcall sub_8F6190(char *a1, unsigned __int8 *a2, int a3)
{
  unsigned int v3; // r12d
  unsigned int v5; // eax
  int v6; // r15d
  char *v7; // rcx
  char *v8; // rsi
  int v9; // eax
  unsigned __int64 v10; // r13
  int v11; // edx
  unsigned __int64 v12; // rsi
  char *v13; // rcx
  int v14; // r15d
  int v15; // edx
  unsigned __int64 v16; // rsi
  __int64 *v17; // rsi
  int v18; // r13d
  __int128 *v19; // r15
  int v20; // eax
  int v21; // r13d
  __int128 *v22; // rax
  int v23; // r13d
  __m128i v24; // xmm3
  __m128i v25; // xmm4
  __int128 *v26; // r15
  __int32 v27; // eax
  __m128i v28; // xmm1
  __m128i v29; // xmm2
  char *v30; // [rsp+8h] [rbp-E8h]
  char *v31; // [rsp+8h] [rbp-E8h]
  int v32; // [rsp+10h] [rbp-E0h]
  int v33; // [rsp+14h] [rbp-DCh]
  int v34; // [rsp+18h] [rbp-D8h]
  int v35; // [rsp+18h] [rbp-D8h]
  __int128 *v36; // [rsp+18h] [rbp-D8h]
  __m128i v37; // [rsp+20h] [rbp-D0h] BYREF
  __m128i v38; // [rsp+30h] [rbp-C0h] BYREF
  __m128i v39; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v40; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v41; // [rsp+60h] [rbp-90h] BYREF
  int v42; // [rsp+68h] [rbp-88h]
  __int64 v43; // [rsp+6Ch] [rbp-84h]
  int v44; // [rsp+74h] [rbp-7Ch]
  __int16 v45; // [rsp+78h] [rbp-78h]
  char v46; // [rsp+7Ah] [rbp-76h]
  int v47; // [rsp+7Ch] [rbp-74h]
  unsigned int v48; // [rsp+80h] [rbp-70h] BYREF
  __int32 v49; // [rsp+84h] [rbp-6Ch]
  int v50; // [rsp+88h] [rbp-68h]
  char *v51; // [rsp+90h] [rbp-60h]
  char *v52; // [rsp+98h] [rbp-58h]
  char *v53; // [rsp+A0h] [rbp-50h]
  char *v54; // [rsp+A8h] [rbp-48h]
  int v55; // [rsp+B0h] [rbp-40h]

  if ( a3 <= 0 || a2 == 0 || !a1 )
    return (unsigned int)-3;
  sub_8F1450(&v48, a2, (char *)&a2[a3]);
  v5 = v48;
  if ( v48 == 2 )
  {
    v6 = 7;
    v38.m128i_i32[3] = 24;
    v37.m128i_i32[0] = 6;
    v33 = 0;
    if ( v55 <= 6 )
      v6 = v55;
    v34 = v50 - v6;
    if ( v50 - v6 < 0 )
    {
      v33 = 1;
      v34 = v6 - v50;
    }
    if ( v50 > 39 )
    {
      v48 = 5;
      v5 = 5;
      goto LABEL_8;
    }
    if ( v50 < -45 )
    {
      v48 = 7;
      v5 = 7;
LABEL_7:
      if ( v5 - 1 <= 1 )
        return 0;
LABEL_8:
      v47 = 24;
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
    if ( v55 > 6 )
      goto LABEL_46;
    if ( v34 <= 10 )
    {
      v48 = 2;
      v32 = 0;
      goto LABEL_37;
    }
    if ( v33 || v34 > 16LL - v55 )
    {
LABEL_46:
      v48 = 1;
      v32 = v34 & 7;
      v34 -= v32;
    }
    else
    {
      v48 = 2;
      v32 = 6 - v55;
      v34 -= 6 - v55;
    }
LABEL_37:
    v7 = v51;
    v8 = v52;
    if ( v51 == v52 )
    {
      v13 = v53;
      if ( v54 == v53 || !v6 )
        goto LABEL_57;
      v10 = 0;
      v9 = 0;
    }
    else
    {
      if ( !v6 )
        goto LABEL_57;
      v9 = 0;
      v10 = 0;
      while ( 1 )
      {
        v11 = *v7++;
        --v6;
        ++v9;
        v10 = v11 - 48 + 10 * v10;
        if ( v7 == v8 )
          break;
        if ( !v6 )
          goto LABEL_59;
        if ( v9 == 9 )
        {
          v30 = v7;
          sub_8EF960(&v37, dword_4F622A0);
          v7 = v30;
          if ( v10 )
          {
            v12 = v10;
            v10 = 0;
            sub_8F0790((__int64)&v37, v12);
            v8 = v52;
            v7 = v30;
          }
          else
          {
            v8 = v52;
          }
          v9 = 0;
        }
      }
      v13 = v53;
      if ( v54 == v53 || !v6 )
      {
LABEL_59:
        sub_8EF960(&v37, &qword_4F62180[4 * v9]);
        if ( v10 )
          sub_8F0790((__int64)&v37, v10);
        if ( v37.m128i_i32[0] != 6 )
        {
          v17 = &qword_4F62180[4 * v32];
          if ( v33 )
          {
            sub_8F0F10(&v37, (__int64)v17);
            if ( v34 > 10 )
            {
              v18 = 0;
              v19 = xmmword_4F62120;
              v20 = v34 >> 3;
              if ( ((v34 >> 3) & 1) != 0 )
                goto LABEL_68;
              while ( 1 )
              {
                v21 = v18 + 1;
                v19 += 2;
                v20 >>= 1;
                if ( !v20 )
                  goto LABEL_17;
                if ( v21 > 1 )
                {
                  v24 = _mm_loadu_si128(&v37);
                  v46 = 0;
                  v45 = 0;
                  v25 = _mm_loadu_si128(&v38);
                  v43 = 0x800000;
                  v26 = &xmmword_4F62120[2 * v21];
                  v44 = 0;
                  v41 = 2;
                  v42 = -125;
                  v47 = 24;
                  v39 = v24;
                  v40 = v25;
                  sub_8F0F10(&v37, (__int64)v26);
                  v33 = 0;
                  while ( sub_8F0EB0(&v37, &v41) )
                  {
                    if ( v37.m128i_i32[0] == 6 )
                      v27 = 23;
                    else
                      v27 = -125 - v37.m128i_i32[2];
                    v33 += v27;
                    v28 = _mm_loadu_si128(&v40);
                    v37 = _mm_loadu_si128(&v39);
                    v37.m128i_i32[2] += v27;
                    v29 = _mm_loadu_si128(&v37);
                    v38 = v28;
                    v39 = v29;
                    sub_8F0F10(&v37, (__int64)v26);
                  }
                  goto LABEL_18;
                }
                v18 = v33;
                if ( (v20 & 1) != 0 )
                {
LABEL_68:
                  v35 = v20;
                  sub_8F0F10(&v37, (__int64)v19);
                  v20 = v35;
                }
              }
            }
            sub_8F0F10(&v37, (__int64)&qword_4F62180[4 * v34]);
            v33 = 0;
          }
          else
          {
            sub_8EF960(&v37, v17);
            v22 = xmmword_4F62120;
            v23 = v34 >> 3;
            if ( v34 > 10 )
            {
              do
              {
                if ( (v23 & 1) != 0 )
                {
                  v36 = v22;
                  sub_8EF960(&v37, v22);
                  v22 = v36;
                }
                v22 += 2;
                v23 >>= 1;
              }
              while ( v23 );
LABEL_17:
              v33 = 0;
              goto LABEL_18;
            }
            sub_8EF960(&v37, &qword_4F62180[4 * v34]);
          }
LABEL_18:
          v5 = v48;
          if ( v49 )
            v37.m128i_i32[1] = v37.m128i_i32[1] == 0;
          if ( v48 != 1 )
            goto LABEL_70;
          if ( (unsigned int)(v37.m128i_i32[0] - 6) <= 1 )
          {
            v37.m128i_i8[14] = 0x80;
            v37.m128i_i64[0] = 2;
            v37.m128i_i32[2] = -148;
            v37.m128i_i16[6] = 0;
            v38.m128i_i32[3] = 24;
          }
          else if ( v37.m128i_i32[0] != 2 )
          {
LABEL_23:
            if ( v37.m128i_i32[2] > 128 )
            {
              v48 = 5;
              v47 = 24;
              HIDWORD(v41) = v49;
LABEL_25:
              LODWORD(v41) = 4;
              v3 = 4;
              goto LABEL_11;
            }
            if ( v37.m128i_i32[2] < -148 )
            {
              v48 = 7;
              v47 = 24;
              HIDWORD(v41) = v49;
LABEL_16:
              LODWORD(v41) = 6;
              v3 = 5;
LABEL_11:
              sub_8EFD90(a1, (int *)&v41);
              return v3;
            }
            v37.m128i_i32[1] = v49;
            v5 = v48;
LABEL_70:
            if ( v5 - 1 <= 1 )
            {
              sub_8EFD90(a1, v37.m128i_i32);
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
        v47 = 24;
        HIDWORD(v41) = 0;
        goto LABEL_16;
      }
    }
    v14 = (_DWORD)v13 + v6;
    do
    {
      if ( v9 == 9 )
      {
        v31 = v13;
        sub_8EF960(&v37, dword_4F622A0);
        v9 = 1;
        v13 = v31;
        if ( v10 )
        {
          v16 = v10;
          v10 = 0;
          sub_8F0790((__int64)&v37, v16);
          v9 = 1;
          v13 = v31;
        }
      }
      else
      {
        ++v9;
        v10 *= 10LL;
      }
      v15 = *v13++;
      v10 += v15 - 48;
    }
    while ( v54 != v13 && v14 != (_DWORD)v13 );
    goto LABEL_59;
  }
LABEL_6:
  if ( v5 )
    goto LABEL_7;
  return (unsigned int)-3;
}
