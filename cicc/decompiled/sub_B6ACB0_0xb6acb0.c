// Function: sub_B6ACB0
// Address: 0xb6acb0
//
__int64 __fastcall sub_B6ACB0(void *s2, size_t n, void *a3, void *a4)
{
  unsigned int v5; // eax
  __m128i si128; // xmm0
  _UNKNOWN **v7; // r8
  __int64 v8; // r13
  __int64 v9; // rcx
  _UNKNOWN **v10; // rbx
  size_t v11; // r14
  size_t v12; // rdx
  int v13; // eax
  unsigned int *v15; // r13
  const char *v16; // rdi
  size_t v17; // rax
  int v18; // eax
  _BYTE *v19; // r12
  _BYTE *v20; // rbx
  void *v21; // r13
  int v22; // eax
  __int64 v23; // rdi
  void *v24; // rax
  void *v25; // rbx
  char *v26; // r12
  __int64 v27; // r14
  __int64 v28; // rax
  const char *v29; // rdi
  size_t v30; // rax
  _UNKNOWN **s2b; // [rsp+8h] [rbp-68h]
  void *s2a; // [rsp+8h] [rbp-68h]
  _UNKNOWN **s2c; // [rsp+8h] [rbp-68h]
  _UNKNOWN **s2d; // [rsp+8h] [rbp-68h]
  void *s1[2]; // [rsp+10h] [rbp-60h] BYREF
  char v36; // [rsp+2Fh] [rbp-41h] BYREF
  __m128i v37; // [rsp+30h] [rbp-40h] BYREF

  s1[0] = a3;
  s1[1] = a4;
  v37.m128i_i64[0] = (__int64)&v36;
  *(_QWORD *)(__readfsqword(0) - 24) = &v37;
  *(_QWORD *)(__readfsqword(0) - 32) = sub_B5B9E0;
  if ( !&_pthread_key_create )
  {
    v5 = -1;
LABEL_46:
    sub_4264C5(v5);
  }
  v5 = pthread_once(&dword_4F818F8, init_routine);
  if ( v5 )
    goto LABEL_46;
  if ( !byte_4F818E8 && (unsigned int)sub_2207590(&byte_4F818E8) )
  {
    qword_4F818F0 = (__int64)&unk_4BB0900;
    sub_2207640(&byte_4F818E8);
  }
  si128 = _mm_load_si128((const __m128i *)s1);
  v37 = si128;
  if ( si128.m128i_i64[1] > 9uLL
    && *(_QWORD *)v37.m128i_i64[0] == 0x69746C6975625F5FLL
    && *(_WORD *)(v37.m128i_i64[0] + 8) == 24430 )
  {
    v37.m128i_i64[1] = si128.m128i_i64[1] - 10;
    s2a = (void *)(v37.m128i_i64[0] + 10);
    v37.m128i_i64[0] += 10;
    v15 = (unsigned int *)sub_B5E340((__int64)&unk_3F3E6B0 - 80, (__int64)&unk_3F3E6B0, (__int64)&v37);
    if ( v15 != (unsigned int *)&unk_3F3E6B0 )
    {
      v16 = (const char *)(qword_4F818F0 + v15[1]);
      if ( v16 )
      {
        v17 = strlen(v16);
        if ( v17 == si128.m128i_i64[1] - 10 && (!v17 || !memcmp(v16, s2a, v17)) )
          return *v15;
      }
      else if ( si128.m128i_i64[1] == 10 )
      {
        return *v15;
      }
    }
  }
  v7 = &off_49794A0;
  v8 = 15;
  do
  {
    while ( 1 )
    {
      v9 = v8 >> 1;
      v10 = &v7[2 * (v8 >> 1)] + 2 * (v8 & 0xFFFFFFFFFFFFFFFELL);
      v11 = (size_t)v10[1];
      v12 = v11;
      if ( n <= v11 )
        v12 = n;
      if ( v12 )
      {
        s2b = v7;
        v13 = memcmp(*v10, s2, v12);
        v7 = s2b;
        v9 = v8 >> 1;
        if ( v13 )
          break;
      }
      if ( n == v11 || n <= v11 )
      {
        v8 = v9;
        goto LABEL_14;
      }
LABEL_6:
      v7 = v10 + 6;
      v8 = v8 - v9 - 1;
      if ( v8 <= 0 )
        goto LABEL_15;
    }
    if ( v13 < 0 )
      goto LABEL_6;
    v8 >>= 1;
LABEL_14:
    ;
  }
  while ( v8 > 0 );
LABEL_15:
  if ( v7 != (_UNKNOWN **)&unk_4979770 && (_UNKNOWN *)n == v7[1] )
  {
    if ( !n || (s2c = v7, v18 = memcmp(*v7, s2, n), v7 = s2c, !v18) )
    {
      v19 = v7[5];
      v20 = s1[1];
      if ( s1[1] >= v19 )
      {
        v21 = s1[0];
        if ( !v19 || (s2d = v7, v22 = memcmp(s1[0], v7[4], (size_t)v7[5]), v7 = s2d, !v22) )
        {
          v23 = (__int64)v7[2];
          v24 = v7[3];
          v25 = (void *)(v20 - v19);
          v26 = &v19[(_QWORD)v21];
          s1[1] = v25;
          v27 = v23 + 8LL * (_QWORD)v24;
          s1[0] = v26;
          v28 = sub_B5E340(v23, v27, (__int64)s1);
          v15 = (unsigned int *)v28;
          if ( v27 != v28 )
          {
            v29 = (const char *)(qword_4F818F0 + *(unsigned int *)(v28 + 4));
            if ( v29 )
            {
              v30 = strlen(v29);
              if ( (void *)v30 == v25 && (!v30 || !memcmp(v29, v26, v30)) )
                return *v15;
            }
            else if ( !v25 )
            {
              return *v15;
            }
          }
        }
      }
    }
  }
  return 0;
}
