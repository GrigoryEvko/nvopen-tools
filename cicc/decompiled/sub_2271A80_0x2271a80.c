// Function: sub_2271A80
// Address: 0x2271a80
//
void __fastcall sub_2271A80(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __m128i si128; // xmm0
  __m128i *v5; // rcx
  __int64 v6; // rsi
  __int64 v7; // rax
  int v8; // eax
  int v9; // r12d
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r9
  __int64 v13; // rcx
  __m128i **v14; // rsi
  __int64 *v15; // r8
  __int64 v16; // rax
  char *v17; // rdi
  char v18; // [rsp-30h] [rbp-100h] BYREF
  __m128i *v19; // [rsp+8h] [rbp-C8h]
  __int64 v20; // [rsp+18h] [rbp-B8h] BYREF
  __int64 v21; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v22; // [rsp+28h] [rbp-A8h] BYREF
  __m128i *v23; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int64 v24; // [rsp+38h] [rbp-98h]
  __m128i v25; // [rsp+40h] [rbp-90h] BYREF
  __int16 v26; // [rsp+50h] [rbp-80h]
  unsigned __int64 v27[2]; // [rsp+60h] [rbp-70h] BYREF
  __m128i v28; // [rsp+70h] [rbp-60h] BYREF
  _DWORD v29[4]; // [rsp+80h] [rbp-50h] BYREF
  __int64 (__fastcall *v30)(_QWORD *, _DWORD *, int); // [rsp+90h] [rbp-40h]
  __int64 (__fastcall *v31)(unsigned int *); // [rsp+98h] [rbp-38h]

  v23 = &v25;
  v19 = &v25;
  v27[0] = 45;
  v3 = sub_22409D0((__int64)&v23, v27, 0);
  v23 = (__m128i *)v3;
  v25.m128i_i64[0] = v27[0];
  *(__m128i *)v3 = _mm_load_si128((const __m128i *)&xmmword_4365B50);
  si128 = _mm_load_si128((const __m128i *)&xmmword_4365C20);
  qmemcpy((void *)(v3 + 32), "EP pipeline: ", 13);
  v5 = v19;
  *(__m128i *)(v3 + 16) = si128;
  v24 = v27[0];
  v23->m128i_i8[v27[0]] = 0;
  v27[0] = (unsigned __int64)&v28;
  if ( v23 == v5 )
  {
    v28 = _mm_load_si128(&v25);
  }
  else
  {
    v27[0] = (unsigned __int64)v23;
    v28.m128i_i64[0] = v25.m128i_i64[0];
  }
  v6 = *a1;
  v29[0] = 1;
  v27[1] = v24;
  v31 = sub_226E290;
  v30 = sub_226EF00;
  sub_235CD80(&v20, v6, a2, qword_4FDA428, qword_4FDA430);
  v7 = v20;
  v20 = 0;
  v21 = v7 | 1;
  if ( (v7 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    if ( v30 )
    {
      v8 = ((__int64 (__fastcall *)(_DWORD *, __int64 *))v31)(v29, &v21);
      v26 = 260;
      v9 = v8;
      v23 = (__m128i *)v27;
      v10 = (__int64 *)sub_CB72A0();
      v13 = 10;
      v14 = &v23;
      v15 = v10;
      v16 = v21;
      v17 = &v18;
      v21 = 0;
      v22 = v16 | 1;
      while ( v13 )
      {
        *(_DWORD *)v17 = *(_DWORD *)v14;
        v14 = (__m128i **)((char *)v14 + 4);
        v17 += 4;
        --v13;
      }
      sub_C63F70((unsigned __int64 *)&v22, v15, v11, 0, (__int64)v15, v12, v18);
      sub_9C66B0(&v22);
      exit(v9);
    }
    sub_4263D6(&v20, v6, v7 | 1);
  }
  if ( v30 )
    v30(v29, v29, 3);
  if ( (__m128i *)v27[0] != &v28 )
    j_j___libc_free_0(v27[0]);
}
