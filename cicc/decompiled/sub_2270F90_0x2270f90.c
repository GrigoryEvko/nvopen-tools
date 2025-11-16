// Function: sub_2270F90
// Address: 0x2270f90
//
void __fastcall sub_2270F90(__int64 *a1, __int64 a2)
{
  __m128i *v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rax
  int v6; // eax
  int v7; // r12d
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r9
  __int64 v11; // rcx
  __m128i **v12; // rsi
  __int64 *v13; // r8
  __int64 v14; // rax
  char *v15; // rdi
  char v16; // [rsp-30h] [rbp-100h] BYREF
  __m128i *v17; // [rsp+8h] [rbp-C8h]
  __int64 v18; // [rsp+18h] [rbp-B8h] BYREF
  __int64 v19; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v20; // [rsp+28h] [rbp-A8h] BYREF
  __m128i *v21; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int64 v22; // [rsp+38h] [rbp-98h]
  __m128i v23; // [rsp+40h] [rbp-90h] BYREF
  __int16 v24; // [rsp+50h] [rbp-80h]
  unsigned __int64 v25[2]; // [rsp+60h] [rbp-70h] BYREF
  __m128i v26; // [rsp+70h] [rbp-60h] BYREF
  _DWORD v27[4]; // [rsp+80h] [rbp-50h] BYREF
  __int64 (__fastcall *v28)(_QWORD *, _DWORD *, int); // [rsp+90h] [rbp-40h]
  __int64 (__fastcall *v29)(unsigned int *); // [rsp+98h] [rbp-38h]

  v21 = &v23;
  v17 = &v23;
  v25[0] = 48;
  v3 = (__m128i *)sub_22409D0((__int64)&v21, v25, 0);
  v21 = v3;
  v23.m128i_i64[0] = v25[0];
  *v3 = _mm_load_si128((const __m128i *)&xmmword_4365B50);
  v3[1] = _mm_load_si128((const __m128i *)&xmmword_4365BB0);
  v3[2] = _mm_load_si128((const __m128i *)&xmmword_4365BC0);
  v22 = v25[0];
  v21->m128i_i8[v25[0]] = 0;
  v25[0] = (unsigned __int64)&v26;
  if ( v21 == &v23 )
  {
    v26 = _mm_load_si128(&v23);
  }
  else
  {
    v25[0] = (unsigned __int64)v21;
    v26.m128i_i64[0] = v23.m128i_i64[0];
  }
  v4 = *a1;
  v27[0] = 1;
  v25[1] = v22;
  v29 = sub_226E290;
  v28 = sub_226EF00;
  sub_2396290(&v18, v4, a2, qword_4FDA328, qword_4FDA330);
  v5 = v18;
  v18 = 0;
  v19 = v5 | 1;
  if ( (v5 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    if ( v28 )
    {
      v6 = ((__int64 (__fastcall *)(_DWORD *, __int64 *))v29)(v27, &v19);
      v24 = 260;
      v7 = v6;
      v21 = (__m128i *)v25;
      v8 = (__int64 *)sub_CB72A0();
      v11 = 10;
      v12 = &v21;
      v13 = v8;
      v14 = v19;
      v15 = &v16;
      v19 = 0;
      v20 = v14 | 1;
      while ( v11 )
      {
        *(_DWORD *)v15 = *(_DWORD *)v12;
        v12 = (__m128i **)((char *)v12 + 4);
        v15 += 4;
        --v11;
      }
      sub_C63F70((unsigned __int64 *)&v20, v13, v9, 0, (__int64)v13, v10, v16);
      sub_9C66B0(&v20);
      exit(v7);
    }
    sub_4263D6(&v18, v4, v5 | 1);
  }
  if ( v28 )
    v28(v27, v27, 3);
  if ( (__m128i *)v25[0] != &v26 )
    j_j___libc_free_0(v25[0]);
}
