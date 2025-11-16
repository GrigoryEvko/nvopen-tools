// Function: sub_8F6930
// Address: 0x8f6930
//
__int64 __fastcall sub_8F6930(char *s, int a2, char *a3)
{
  int v3; // r12d
  char *v4; // rsi
  char *v5; // r13
  int v6; // ecx
  unsigned int v7; // edi
  __int64 i; // rax
  __int8 v9; // dl
  __int8 v10; // dl
  unsigned __int8 v11; // al
  __int8 v12; // di
  unsigned __int8 v13; // cl
  int v14; // edx
  int v15; // eax
  int v16; // ebx
  __int32 v17; // eax
  __int64 result; // rax
  __int64 v19; // r9
  char v20; // di
  int v21; // eax
  int v22; // r10d
  int v23; // r11d
  unsigned __int64 v24; // rax
  int v25; // ecx
  __m128i v26; // xmm5
  signed __int64 v27; // rcx
  __int64 *v28; // rcx
  __int64 v29; // rcx
  const __m128i *v30; // rax
  __m128i si128; // xmm7
  __m128i v32; // xmm1
  int v33; // eax
  __m128i v34; // xmm3
  __int64 v35; // rax
  int v36; // edx
  int v37; // [rsp-108h] [rbp-108h]
  char v38; // [rsp-108h] [rbp-108h]
  int v39; // [rsp-104h] [rbp-104h]
  __int64 j; // [rsp-100h] [rbp-100h]
  __m128i v41; // [rsp-F8h] [rbp-F8h] BYREF
  __m128i v42; // [rsp-E8h] [rbp-E8h] BYREF
  __m128i v43; // [rsp-D8h] [rbp-D8h] BYREF
  __m128i v44; // [rsp-C8h] [rbp-C8h] BYREF
  __m128i v45; // [rsp-B8h] [rbp-B8h] BYREF
  __m128i v46; // [rsp-A8h] [rbp-A8h] BYREF
  __m128i v47; // [rsp-98h] [rbp-98h] BYREF
  __m128i v48; // [rsp-88h] [rbp-88h]
  _DWORD v49[30]; // [rsp-78h] [rbp-78h] BYREF

  if ( !s )
    return 4294967294LL;
  v3 = a2;
  if ( a2 <= 0 )
    return 4294967294LL;
  v4 = a3;
  if ( !a3 )
    return 4294967293LL;
  v5 = s;
  v6 = 0;
  v42.m128i_i8[2] = 0;
  v7 = dword_4F07580;
  for ( i = 0; i != 6; ++i )
  {
    while ( !v7 )
    {
      v10 = v4[-i + 7];
      v41.m128i_i8[i + 12] = v10;
      if ( v10 )
        v6 = 1;
      if ( ++i == 6 )
        goto LABEL_12;
    }
    v9 = v4[i];
    v41.m128i_i8[i + 12] = v9;
    if ( v9 )
      v6 = 1;
  }
LABEL_12:
  if ( v7 )
  {
    v11 = v4[6];
    v12 = v11 & 0xF;
    v42.m128i_i8[2] = v11 & 0xF | 0x10;
    if ( (v11 & 0xF) != 0 )
    {
      v13 = v4[7];
      v14 = v13 >> 7;
      v41.m128i_i32[1] = v14;
      v15 = (16 * (v13 & 0x7F)) | (v11 >> 4);
      if ( v15 != 2047 )
      {
        if ( !v15 )
          goto LABEL_16;
LABEL_31:
        v41.m128i_i32[0] = 2;
        v17 = v15 - 1022;
        goto LABEL_22;
      }
LABEL_32:
      v41.m128i_i32[0] = 3;
      v17 = 1025;
      goto LABEL_22;
    }
    v4 += 7;
  }
  else
  {
    v11 = v4[1];
    v12 = v11 & 0xF;
    v42.m128i_i8[2] = v11 & 0xF | 0x10;
    if ( (v11 & 0xF) != 0 )
    {
      v15 = (16 * (*v4 & 0x7F)) | (v11 >> 4);
      v14 = (unsigned __int8)*v4 >> 7;
      v6 = 1;
      v41.m128i_i32[1] = v14;
      if ( v15 == 2047 )
        goto LABEL_32;
      goto LABEL_19;
    }
  }
  v20 = *v4;
  v14 = (unsigned __int8)*v4 >> 7;
  v41.m128i_i32[1] = v14;
  v15 = (16 * (v20 & 0x7F)) | (v11 >> 4);
  v12 = 0;
  if ( v15 == 2047 )
  {
    if ( !v6 )
    {
      v41.m128i_i32[0] = 4;
      v17 = 1025;
      goto LABEL_22;
    }
    goto LABEL_32;
  }
LABEL_19:
  if ( v15 )
    goto LABEL_31;
  if ( !v6 )
  {
    v41.m128i_i32[0] = 6;
    v17 = -1022;
    goto LABEL_22;
  }
LABEL_16:
  v42.m128i_i8[2] = v12;
  v16 = sub_8EE4D0(&v41.m128i_i8[12], 53);
  sub_8EE880(&v41.m128i_i8[12], 53, v16);
  v14 = v41.m128i_i32[1];
  v41.m128i_i32[0] = 2;
  v17 = -1021 - v16;
LABEL_22:
  v41.m128i_i32[2] = v17;
  v42.m128i_i32[3] = 53;
  if ( v14 && v41.m128i_i32[0] != 3 )
  {
    *v5 = 45;
    --v3;
    ++v5;
  }
  result = sub_8F1A40(v5, v3, v41.m128i_i32);
  if ( (_DWORD)result == -1 )
  {
    if ( v41.m128i_i32[2] <= 0 )
    {
      if ( v41.m128i_i32[2] < -1021 )
      {
        v25 = -1021 - v41.m128i_i32[2];
        goto LABEL_37;
      }
    }
    else
    {
      v21 = sub_8EE460((__int64)&v41.m128i_i64[1] + 4, v42.m128i_i32[3]);
      if ( v23 >= v22 - v21 )
      {
        v24 = 30103LL * v23;
        if ( (__int64)v24 <= 2299999 )
        {
          v26 = _mm_loadu_si128(&v42);
          v43 = _mm_loadu_si128(&v41);
          v44 = v26;
          v43.m128i_i32[1] = 0;
          v27 = v24 / 0x186A0;
          v39 = v24 / 0x186A0;
          if ( v24 / 0x186A0 )
          {
            v28 = &qword_4F61E40[4 * (int)v27];
            do
            {
              if ( !sub_8F0EB0(&v43, v28) )
              {
                v27 = v39;
                goto LABEL_46;
              }
              v28 = (__int64 *)(v29 - 32);
              --v39;
            }
            while ( v39 );
            v27 = 0;
          }
          else
          {
            v39 = 0;
          }
LABEL_46:
          if ( 30103LL * v22 / 100000 > v27 )
          {
            v30 = (const __m128i *)&qword_4F61E40[4 * v39];
            si128 = _mm_load_si128(v30 + 1);
            v45 = _mm_load_si128(v30);
            v46 = si128;
            if ( v43.m128i_i32[0] == 6 )
            {
              v37 = 0;
            }
            else
            {
              for ( j = 0; ; ++j )
              {
                v32 = _mm_loadu_si128(&v44);
                v47 = _mm_loadu_si128(&v43);
                v48 = v32;
                sub_8F0F10(&v47, (__int64)&v45);
                v33 = sub_8EEF20((__int64)&v47);
                v34 = _mm_loadu_si128(&v46);
                v38 = v33;
                v47 = _mm_loadu_si128(&v45);
                v48 = v34;
                if ( v33 )
                {
                  if ( v47.m128i_i32[0] != 6 )
                    sub_8F06E0(&v47, v33);
                }
                else
                {
                  v47.m128i_i32[0] = 6;
                }
                if ( sub_8F0EB0(&v43, &v47) )
                {
                  --v38;
                  sub_8EF5D0(&v47, &v45);
                }
                sub_8EF5D0(&v43, &v47);
                *((_BYTE *)&v49[2] + j) = v38 + 48;
                v37 = j + 1;
                if ( v43.m128i_i32[0] == 6 )
                  break;
                sub_8F06E0(&v43, 10);
                if ( v43.m128i_i32[0] == 6 )
                  break;
              }
            }
            v35 = v37;
            do
            {
              v36 = v35;
              if ( (int)v35 <= 1 )
                break;
              --v35;
            }
            while ( *((_BYTE *)&v49[2] + v35) == 48 );
            v49[1] = v39 + 1;
            v49[12] = v36;
            v49[0] = v41.m128i_i32[1];
            *((_BYTE *)&v49[2] + v36) = 0;
            return sub_8EFB80(v5, v3, (__int64)v49);
          }
        }
      }
    }
    v25 = 0;
LABEL_37:
    sub_8F2910((__int64)v49, v19, 0x7FFFFFFF, v25);
    return sub_8EFB80(v5, v3, (__int64)v49);
  }
  return result;
}
