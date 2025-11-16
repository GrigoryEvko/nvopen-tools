// Function: sub_18C55D0
// Address: 0x18c55d0
//
void __fastcall sub_18C55D0(__int64 a1)
{
  int v2; // r9d
  __int64 v3; // rax
  __int64 *v4; // rdi
  __int64 *v5; // rbx
  _BYTE *v6; // r8
  unsigned __int64 v7; // rcx
  unsigned int v8; // edx
  __m128i *v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // r8
  _BYTE *v13; // rsi
  __m128i *v14; // rdi
  _BYTE *v15; // [rsp+18h] [rbp-1B8h]
  unsigned __int64 v16; // [rsp+20h] [rbp-1B0h]
  __int64 *v17; // [rsp+28h] [rbp-1A8h]
  _QWORD *v18; // [rsp+30h] [rbp-1A0h] BYREF
  char v19; // [rsp+40h] [rbp-190h]
  __m128i *v20; // [rsp+50h] [rbp-180h] BYREF
  __int64 v21; // [rsp+58h] [rbp-178h]
  __m128i v22; // [rsp+60h] [rbp-170h] BYREF
  __m128i *v23; // [rsp+70h] [rbp-160h] BYREF
  __int64 v24; // [rsp+78h] [rbp-158h]
  __m128i v25; // [rsp+80h] [rbp-150h] BYREF
  __int64 *v26; // [rsp+90h] [rbp-140h] BYREF
  __int64 v27; // [rsp+98h] [rbp-138h]
  _WORD v28[152]; // [rsp+A0h] [rbp-130h] BYREF

  v28[0] = 260;
  v26 = &qword_4FADE20;
  sub_16C2DE0((__int64)&v18, (__int64)&v26, 0xFFFFFFFFFFFFFFFFLL, 1, 0);
  if ( (v19 & 1) != 0 && (_DWORD)v18 )
    sub_16BD130("BlockExtractor couldn't load the file.", 1u);
  v26 = (__int64 *)v28;
  v27 = 0x1000000000LL;
  v3 = v18[2];
  v20 = (__m128i *)v18[1];
  v21 = v3 - (_QWORD)v20;
  sub_16D2880((char **)&v20, (__int64)&v26, 10, -1, 0, v2);
  v4 = v26;
  v17 = &v26[2 * (unsigned int)v27];
  if ( v17 != v26 )
  {
    v5 = v26;
    while ( 1 )
    {
      LOBYTE(v20) = 32;
      v10 = sub_16D20C0(v5, (char *)&v20, 1u, 0);
      if ( v10 == -1 )
      {
        v13 = (_BYTE *)*v5;
        v10 = v5[1];
        v7 = 0;
        v6 = 0;
      }
      else
      {
        v11 = v5[1];
        v12 = v10 + 1;
        v13 = (_BYTE *)*v5;
        if ( v10 + 1 > v11 )
          v12 = v5[1];
        v7 = v5[1] - v12;
        v6 = &v13[v12];
        if ( v10 && v10 > v11 )
          v10 = v5[1];
      }
      if ( v13 )
      {
        v15 = v6;
        v16 = v7;
        v20 = &v22;
        sub_18C4AE0((__int64 *)&v20, v13, (__int64)&v13[v10]);
        v6 = v15;
        v23 = &v25;
        v7 = v16;
        if ( !v15 )
          goto LABEL_26;
      }
      else
      {
        v20 = &v22;
        v21 = 0;
        v22.m128i_i8[0] = 0;
        v23 = &v25;
        if ( !v6 )
        {
LABEL_26:
          v24 = 0;
          v25.m128i_i8[0] = 0;
          goto LABEL_7;
        }
      }
      sub_18C4AE0((__int64 *)&v23, v6, (__int64)&v6[v7]);
LABEL_7:
      v8 = *(_DWORD *)(a1 + 320);
      if ( v8 >= *(_DWORD *)(a1 + 324) )
      {
        sub_18C53B0(a1 + 312, 0);
        v8 = *(_DWORD *)(a1 + 320);
      }
      v9 = (__m128i *)(*(_QWORD *)(a1 + 312) + ((unsigned __int64)v8 << 6));
      if ( v9 )
      {
        v9->m128i_i64[0] = (__int64)v9[1].m128i_i64;
        if ( v20 == &v22 )
        {
          v9[1] = _mm_load_si128(&v22);
        }
        else
        {
          v9->m128i_i64[0] = (__int64)v20;
          v9[1].m128i_i64[0] = v22.m128i_i64[0];
        }
        v9->m128i_i64[1] = v21;
        v20 = &v22;
        v21 = 0;
        v22.m128i_i8[0] = 0;
        v9[2].m128i_i64[0] = (__int64)v9[3].m128i_i64;
        if ( v23 == &v25 )
        {
          v9[3] = _mm_load_si128(&v25);
        }
        else
        {
          v9[2].m128i_i64[0] = (__int64)v23;
          v9[3].m128i_i64[0] = v25.m128i_i64[0];
        }
        v9[2].m128i_i64[1] = v24;
        ++*(_DWORD *)(a1 + 320);
      }
      else
      {
        v14 = v23;
        *(_DWORD *)(a1 + 320) = v8 + 1;
        if ( v14 != &v25 )
          j_j___libc_free_0(v14, v25.m128i_i64[0] + 1);
      }
      if ( v20 != &v22 )
        j_j___libc_free_0(v20, v22.m128i_i64[0] + 1);
      v5 += 2;
      if ( v17 == v5 )
      {
        v4 = v26;
        break;
      }
    }
  }
  if ( v4 != (__int64 *)v28 )
    _libc_free((unsigned __int64)v4);
  if ( (v19 & 1) == 0 )
  {
    if ( v18 )
      (*(void (__fastcall **)(_QWORD *))(*v18 + 8LL))(v18);
  }
}
