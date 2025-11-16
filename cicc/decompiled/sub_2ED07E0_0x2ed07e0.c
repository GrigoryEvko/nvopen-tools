// Function: sub_2ED07E0
// Address: 0x2ed07e0
//
_QWORD *__fastcall sub_2ED07E0(_QWORD *a1, const char *a2, const char *a3, __int64 a4)
{
  __int64 v4; // r15
  size_t v7; // rax
  __int64 v8; // r9
  size_t v9; // rax
  _QWORD *result; // rax
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 (__fastcall *v13)(__int64, const char *, __int64, __int64, const char *); // rax
  __int64 v14; // rdx
  const __m128i *v15; // rbx
  unsigned __int64 v16; // rcx
  __int64 v17; // r8
  unsigned __int64 v18; // rsi
  int v19; // eax
  __m128i *v20; // rdx
  __m128i v21; // xmm1
  __int64 v22; // rdi
  __int64 v23; // rdi
  char *v24; // rbx
  __int64 v25; // [rsp+8h] [rbp-78h]
  __int64 v26; // [rsp+8h] [rbp-78h]
  _QWORD v27[6]; // [rsp+10h] [rbp-70h] BYREF
  char v28; // [rsp+40h] [rbp-40h]

  v4 = 0;
  *a1 = 0;
  a1[1] = a2;
  if ( a2 )
  {
    v25 = a4;
    v7 = strlen(a2);
    a4 = v25;
    v4 = v7;
  }
  a1[2] = v4;
  v8 = 0;
  a1[3] = a3;
  if ( a3 )
  {
    v26 = a4;
    v9 = strlen(a3);
    a4 = v26;
    v8 = v9;
  }
  result = qword_5021050;
  a1[4] = v8;
  a1[5] = a4;
  v11 = qword_5021050[0];
  v12 = qword_5021050[2];
  qword_5021050[0] = a1;
  *a1 = v11;
  if ( v12 )
  {
    v13 = *(__int64 (__fastcall **)(__int64, const char *, __int64, __int64, const char *))(*(_QWORD *)v12 + 24LL);
    if ( (char *)v13 == (char *)sub_2ED0310 )
    {
      v14 = *(unsigned int *)(v12 + 32);
      v27[2] = a3;
      v15 = (const __m128i *)v27;
      v27[5] = a4;
      v16 = *(unsigned int *)(v12 + 36);
      v17 = v14 + 1;
      v27[0] = a2;
      v27[1] = v4;
      v18 = *(_QWORD *)(v12 + 24);
      v27[4] = &unk_4A29C80;
      v19 = v14;
      v27[3] = v8;
      v28 = 1;
      if ( v14 + 1 > v16 )
      {
        v23 = v12 + 24;
        if ( v18 > (unsigned __int64)v27 || (unsigned __int64)v27 >= v18 + 56 * v14 )
        {
          sub_2ED0250(v23, v14 + 1, v14, v16, v17, v8);
          v14 = *(unsigned int *)(v12 + 32);
          v18 = *(_QWORD *)(v12 + 24);
          v19 = *(_DWORD *)(v12 + 32);
        }
        else
        {
          v24 = (char *)v27 - v18;
          sub_2ED0250(v23, v14 + 1, v14, v16, v17, v8);
          v18 = *(_QWORD *)(v12 + 24);
          v14 = *(unsigned int *)(v12 + 32);
          v15 = (const __m128i *)&v24[v18];
          v19 = *(_DWORD *)(v12 + 32);
        }
      }
      v20 = (__m128i *)(v18 + 56 * v14);
      if ( v20 )
      {
        v21 = _mm_loadu_si128(v15 + 1);
        *v20 = _mm_loadu_si128(v15);
        v20[1] = v21;
        v20[2].m128i_i64[1] = v15[2].m128i_i64[1];
        v20[3].m128i_i8[0] = v15[3].m128i_i8[0];
        v20[2].m128i_i64[0] = (__int64)&unk_4A29C80;
        v19 = *(_DWORD *)(v12 + 32);
      }
      v22 = *(_QWORD *)(v12 + 16);
      *(_DWORD *)(v12 + 32) = v19 + 1;
      return sub_C52F90(v22, (__int64)a2, v4);
    }
    else
    {
      return (_QWORD *)v13(v12, a2, v4, a4, a3);
    }
  }
  return result;
}
