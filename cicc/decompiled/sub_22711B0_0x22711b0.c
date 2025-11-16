// Function: sub_22711B0
// Address: 0x22711b0
//
void __fastcall sub_22711B0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __m128i si128; // xmm0
  __int64 v5; // rsi
  __int64 v6; // rax
  int v7; // eax
  int v8; // r12d
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r9
  __int64 v12; // rcx
  __m128i **v13; // rsi
  __int64 *v14; // r8
  __int64 v15; // rax
  char *v16; // rdi
  char v17; // [rsp-30h] [rbp-100h] BYREF
  __m128i *v18; // [rsp+8h] [rbp-C8h]
  __int64 v19; // [rsp+18h] [rbp-B8h] BYREF
  __int64 v20; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v21; // [rsp+28h] [rbp-A8h] BYREF
  __m128i *v22; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int64 v23; // [rsp+38h] [rbp-98h]
  __m128i v24; // [rsp+40h] [rbp-90h] BYREF
  __int16 v25; // [rsp+50h] [rbp-80h]
  unsigned __int64 v26[2]; // [rsp+60h] [rbp-70h] BYREF
  __m128i v27; // [rsp+70h] [rbp-60h] BYREF
  _DWORD v28[4]; // [rsp+80h] [rbp-50h] BYREF
  __int64 (__fastcall *v29)(_QWORD *, _DWORD *, int); // [rsp+90h] [rbp-40h]
  __int64 (__fastcall *v30)(unsigned int *); // [rsp+98h] [rbp-38h]

  v22 = &v24;
  v18 = &v24;
  v26[0] = 50;
  v3 = sub_22409D0((__int64)&v22, v26, 0);
  v22 = (__m128i *)v3;
  v24.m128i_i64[0] = v26[0];
  *(__m128i *)v3 = _mm_load_si128((const __m128i *)&xmmword_4365B50);
  si128 = _mm_load_si128((const __m128i *)&xmmword_4365BD0);
  *(_WORD *)(v3 + 48) = 8250;
  *(__m128i *)(v3 + 16) = si128;
  *(__m128i *)(v3 + 32) = _mm_load_si128((const __m128i *)&xmmword_4365BE0);
  v23 = v26[0];
  v22->m128i_i8[v26[0]] = 0;
  v26[0] = (unsigned __int64)&v27;
  if ( v22 == &v24 )
  {
    v27 = _mm_load_si128(&v24);
  }
  else
  {
    v26[0] = (unsigned __int64)v22;
    v27.m128i_i64[0] = v24.m128i_i64[0];
  }
  v5 = *a1;
  v28[0] = 1;
  v26[1] = v23;
  v30 = sub_226E290;
  v29 = sub_226EF00;
  sub_235CD80(&v19, v5, a2, qword_4FDA528, qword_4FDA530);
  v6 = v19;
  v19 = 0;
  v20 = v6 | 1;
  if ( (v6 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    if ( v29 )
    {
      v7 = ((__int64 (__fastcall *)(_DWORD *, __int64 *))v30)(v28, &v20);
      v25 = 260;
      v8 = v7;
      v22 = (__m128i *)v26;
      v9 = (__int64 *)sub_CB72A0();
      v12 = 10;
      v13 = &v22;
      v14 = v9;
      v15 = v20;
      v16 = &v17;
      v20 = 0;
      v21 = v15 | 1;
      while ( v12 )
      {
        *(_DWORD *)v16 = *(_DWORD *)v13;
        v13 = (__m128i **)((char *)v13 + 4);
        v16 += 4;
        --v12;
      }
      sub_C63F70((unsigned __int64 *)&v21, v14, v10, 0, (__int64)v14, v11, v17);
      sub_9C66B0(&v21);
      exit(v8);
    }
    sub_4263D6(&v19, v5, v6 | 1);
  }
  if ( v29 )
    v29(v28, v28, 3);
  if ( (__m128i *)v26[0] != &v27 )
    j_j___libc_free_0(v26[0]);
}
