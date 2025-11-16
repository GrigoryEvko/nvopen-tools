// Function: sub_2270910
// Address: 0x2270910
//
void __fastcall sub_2270910(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __m128i si128; // xmm0
  __m128i *v5; // rcx
  __m128i v6; // xmm0
  __int64 v7; // rsi
  __int64 v8; // rax
  int v9; // eax
  int v10; // r12d
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r9
  __int64 v14; // rcx
  __m128i **v15; // rsi
  __int64 *v16; // r8
  __int64 v17; // rax
  char *v18; // rdi
  char v19; // [rsp-30h] [rbp-100h] BYREF
  __m128i *v20; // [rsp+8h] [rbp-C8h]
  __int64 v21; // [rsp+18h] [rbp-B8h] BYREF
  __int64 v22; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v23; // [rsp+28h] [rbp-A8h] BYREF
  __m128i *v24; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int64 v25; // [rsp+38h] [rbp-98h]
  __m128i v26; // [rsp+40h] [rbp-90h] BYREF
  __int16 v27; // [rsp+50h] [rbp-80h]
  unsigned __int64 v28[2]; // [rsp+60h] [rbp-70h] BYREF
  __m128i v29; // [rsp+70h] [rbp-60h] BYREF
  _DWORD v30[4]; // [rsp+80h] [rbp-50h] BYREF
  __int64 (__fastcall *v31)(_QWORD *, _DWORD *, int); // [rsp+90h] [rbp-40h]
  __int64 (__fastcall *v32)(unsigned int *); // [rsp+98h] [rbp-38h]

  v24 = &v26;
  v20 = &v26;
  v28[0] = 57;
  v3 = sub_22409D0((__int64)&v24, v28, 0);
  v24 = (__m128i *)v3;
  v26.m128i_i64[0] = v28[0];
  *(__m128i *)v3 = _mm_load_si128((const __m128i *)&xmmword_4365B50);
  si128 = _mm_load_si128((const __m128i *)&xmmword_4365B70);
  *(_QWORD *)(v3 + 48) = 0x3A656E696C657069LL;
  v5 = v20;
  *(__m128i *)(v3 + 16) = si128;
  v6 = _mm_load_si128((const __m128i *)&xmmword_4365B80);
  *(_BYTE *)(v3 + 56) = 32;
  *(__m128i *)(v3 + 32) = v6;
  v25 = v28[0];
  v24->m128i_i8[v28[0]] = 0;
  v28[0] = (unsigned __int64)&v29;
  if ( v24 == v5 )
  {
    v29 = _mm_load_si128(&v26);
  }
  else
  {
    v28[0] = (unsigned __int64)v24;
    v29.m128i_i64[0] = v26.m128i_i64[0];
  }
  v7 = *a1;
  v30[0] = 1;
  v28[1] = v25;
  v32 = sub_226E290;
  v31 = sub_226EF00;
  sub_2394710(&v21, v7, a2, qword_4FD9A28, qword_4FD9A30);
  v8 = v21;
  v21 = 0;
  v22 = v8 | 1;
  if ( (v8 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    if ( v31 )
    {
      v9 = ((__int64 (__fastcall *)(_DWORD *, __int64 *))v32)(v30, &v22);
      v27 = 260;
      v10 = v9;
      v24 = (__m128i *)v28;
      v11 = (__int64 *)sub_CB72A0();
      v14 = 10;
      v15 = &v24;
      v16 = v11;
      v17 = v22;
      v18 = &v19;
      v22 = 0;
      v23 = v17 | 1;
      while ( v14 )
      {
        *(_DWORD *)v18 = *(_DWORD *)v15;
        v15 = (__m128i **)((char *)v15 + 4);
        v18 += 4;
        --v14;
      }
      sub_C63F70((unsigned __int64 *)&v23, v16, v12, 0, (__int64)v16, v13, v19);
      sub_9C66B0(&v23);
      exit(v10);
    }
    sub_4263D6(&v21, v7, v8 | 1);
  }
  if ( v31 )
    v31(v30, v30, 3);
  if ( (__m128i *)v28[0] != &v29 )
    j_j___libc_free_0(v28[0]);
}
