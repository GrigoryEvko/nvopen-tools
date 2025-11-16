// Function: sub_CA4820
// Address: 0xca4820
//
_QWORD *__fastcall sub_CA4820(_QWORD *a1, __m128i *a2, const __m128i *a3, __int64 a4)
{
  bool v6; // zf
  __m128i *v7; // r12
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rsi
  volatile signed __int32 *v12; // rdi
  char v13; // cl
  char *v14; // rdx
  int v15; // eax
  unsigned __int64 *v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rsi
  int v20; // eax
  __int64 v21; // rdx
  _BYTE *v22; // rsi
  size_t v24; // rax
  int v25; // [rsp+4h] [rbp-1CCh]
  char v26; // [rsp+4h] [rbp-1CCh]
  char v27; // [rsp+8h] [rbp-1C8h]
  __m128i *v28; // [rsp+10h] [rbp-1C0h] BYREF
  __int64 v29; // [rsp+18h] [rbp-1B8h]
  __m128i v30; // [rsp+20h] [rbp-1B0h] BYREF
  char *s[2]; // [rsp+30h] [rbp-1A0h] BYREF
  __m128i v32; // [rsp+40h] [rbp-190h]
  __int64 v33; // [rsp+50h] [rbp-180h]
  char *v34; // [rsp+60h] [rbp-170h] BYREF
  char *v35; // [rsp+68h] [rbp-168h]
  __int64 v36; // [rsp+70h] [rbp-160h]
  _BYTE v37[136]; // [rsp+78h] [rbp-158h] BYREF
  __m128i v38; // [rsp+100h] [rbp-D0h] BYREF
  __m128i v39; // [rsp+110h] [rbp-C0h] BYREF
  __int128 v40; // [rsp+120h] [rbp-B0h]
  __int128 v41; // [rsp+130h] [rbp-A0h]
  __int128 v42; // [rsp+140h] [rbp-90h]
  int v43; // [rsp+150h] [rbp-80h]
  int v44; // [rsp+154h] [rbp-7Ch]

  v6 = a2[20].m128i_i8[8] == 0;
  v34 = v37;
  v35 = 0;
  v36 = 128;
  if ( v6 || (v7 = a2, (a2[20].m128i_i8[0] & 1) != 0) )
  {
    v33 = a3[2].m128i_i64[0];
    *(__m128i *)s = _mm_loadu_si128(a3);
    v32 = _mm_loadu_si128(a3 + 1);
  }
  else
  {
    sub_CA0EC0((__int64)a3, (__int64)&v34);
    a2 = (__m128i *)&v34;
    LOWORD(v40) = 261;
    v38 = *(__m128i *)((char *)v7 + 168);
    sub_C846B0((__int64)&v38, (unsigned __int8 **)&v34);
    LOWORD(v33) = 261;
    s[0] = v34;
    s[1] = v35;
  }
  v8 = sub_22077B0(88);
  v9 = v8;
  if ( v8 )
  {
    *(_QWORD *)(v8 + 8) = 0x100000001LL;
    *(_QWORD *)v8 = off_4979CD8;
    *(_QWORD *)(v8 + 24) = v8 + 40;
    *(_QWORD *)(v8 + 32) = 0;
    *(_BYTE *)(v8 + 40) = 0;
    *(_DWORD *)(v8 + 56) = 9;
    *(_QWORD *)(v8 + 16) = off_4979CB0;
    *(_QWORD *)(v8 + 64) = 0;
    *(_QWORD *)(v8 + 72) = 0;
    *(_BYTE *)(v8 + 80) = 1;
    v10 = sub_22077B0(112);
    if ( v10 )
    {
      v11 = v10 + 16;
      *(_QWORD *)(v10 + 8) = 0x100000001LL;
      *(_QWORD *)v10 = &unk_49DCA38;
      memset((void *)(v10 + 16), 0, 0x60u);
      *(_BYTE *)(v10 + 60) = 1;
      *(_QWORD *)(v10 + 24) = v10 + 40;
      *(_DWORD *)(v10 + 56) = 9;
      *(_DWORD *)(v10 + 108) = 0xFFFF;
    }
    else
    {
      v11 = 16;
    }
    v12 = *(volatile signed __int32 **)(v9 + 72);
    *(_QWORD *)(v9 + 64) = v11;
    *(_QWORD *)(v9 + 72) = v10;
    if ( v12 )
      sub_A191D0(v12);
    v38.m128i_i64[1] = 0;
    v38.m128i_i64[0] = (__int64)&v39.m128i_i64[1];
    v13 = *(_BYTE *)(v9 + 80);
    v39.m128i_i64[0] = 128;
    if ( BYTE1(v33) == 1 )
    {
      if ( (_BYTE)v33 == 1 )
      {
        v14 = 0;
        a2 = 0;
        goto LABEL_11;
      }
      if ( (unsigned __int8)(v33 - 3) <= 3u )
      {
        if ( (_BYTE)v33 == 4 )
        {
          a2 = *(__m128i **)s[0];
          v14 = (char *)*((_QWORD *)s[0] + 1);
          goto LABEL_11;
        }
        if ( (unsigned __int8)v33 <= 4u )
        {
          if ( (_BYTE)v33 == 3 )
          {
            a2 = (__m128i *)s[0];
            v14 = 0;
            if ( s[0] )
            {
              v26 = v13;
              v24 = strlen(s[0]);
              a2 = (__m128i *)s[0];
              v13 = v26;
              v14 = (char *)v24;
            }
            goto LABEL_11;
          }
        }
        else if ( (unsigned __int8)(v33 - 5) <= 1u )
        {
          v14 = s[1];
          a2 = (__m128i *)s[0];
          goto LABEL_11;
        }
        BUG();
      }
    }
    v27 = v13;
    sub_CA0EC0((__int64)s, (__int64)&v38);
    v14 = (char *)v38.m128i_i64[1];
    a2 = (__m128i *)v38.m128i_i64[0];
    v13 = v27;
LABEL_11:
    v15 = sub_C82F00(*(_QWORD *)(v9 + 64), a2, (size_t)v14, v13);
    v16 = (unsigned __int64 *)v38.m128i_i64[0];
    *(_DWORD *)a4 = v15;
    *(_QWORD *)(a4 + 8) = v17;
    if ( v16 != &v39.m128i_u64[1] )
      _libc_free(v16, a2);
    v18 = *(_QWORD *)(v9 + 64);
    if ( v18 )
    {
      v43 = 0;
      v40 = 0;
      v38.m128i_i64[0] = (__int64)&v39;
      v38.m128i_i64[1] = 0;
      v39.m128i_i8[0] = 0;
      LODWORD(v40) = 9;
      BYTE4(v40) = 1;
      v44 = 0xFFFF;
      v41 = 0;
      v42 = 0;
      if ( *(_QWORD *)(v18 + 16) )
      {
        sub_2240A30(&v38);
        v19 = *(_QWORD *)(v9 + 64);
        v20 = *(_DWORD *)(v19 + 40);
        if ( v20 == 9 )
        {
          sub_C832B0(&v38, v19 + 8);
          v20 = 9;
          v19 = *(_QWORD *)(v9 + 64);
          if ( (v41 & 1) == 0 )
            v20 = DWORD2(v40);
        }
        v25 = v20;
        v21 = *(_QWORD *)(v19 + 8) + *(_QWORD *)(v19 + 16);
        v22 = *(_BYTE **)(v19 + 8);
        v28 = &v30;
        sub_CA1F00((__int64 *)&v28, v22, v21);
        v38.m128i_i64[0] = (__int64)&v39;
        if ( v28 == &v30 )
        {
          v39 = _mm_load_si128(&v30);
        }
        else
        {
          v38.m128i_i64[0] = (__int64)v28;
          v39.m128i_i64[0] = v30.m128i_i64[0];
        }
        a2 = &v38;
        v28 = &v30;
        v38.m128i_i64[1] = v29;
        v29 = 0;
        v30.m128i_i8[0] = 0;
        LODWORD(v40) = v25;
        sub_2240D70(v9 + 24, &v38);
        *(_DWORD *)(v9 + 56) = v40;
        sub_2240A30(&v38);
        if ( v28 != &v30 )
        {
          a2 = (__m128i *)(v30.m128i_i64[0] + 1);
          j_j___libc_free_0(v28, v30.m128i_i64[0] + 1);
        }
      }
      else
      {
        sub_2240A30(&v38);
      }
    }
    v6 = *(_QWORD *)(v9 + 32) == 0;
    a1[1] = v9;
    *a1 = v9 + 16;
    if ( v6 )
    {
      *a1 = 0;
      a1[1] = 0;
      sub_A191D0((volatile signed __int32 *)v9);
    }
    goto LABEL_22;
  }
  *a1 = 16;
  a1[1] = 0;
  if ( !MEMORY[0x20] )
    *a1 = 0;
LABEL_22:
  if ( v34 != v37 )
    _libc_free(v34, a2);
  return a1;
}
