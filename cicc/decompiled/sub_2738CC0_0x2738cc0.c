// Function: sub_2738CC0
// Address: 0x2738cc0
//
unsigned __int64 __fastcall sub_2738CC0(__int64 a1, int *a2, __int64 *a3, __int64 *a4)
{
  __int64 v4; // rbp
  unsigned __int64 v6; // rcx
  int v7; // r11d
  unsigned __int64 v8; // r8
  unsigned __int64 v9; // rsi
  __int64 v10; // r10
  __int64 v11; // r9
  unsigned __int64 v12; // rdx
  int v13; // eax
  unsigned __int64 result; // rax
  const __m128i *v15; // rbx
  unsigned __int64 v16; // r9
  __int64 v17; // r8
  char *v18; // rbx
  int v19; // [rsp-38h] [rbp-38h] BYREF
  char v20; // [rsp-34h] [rbp-34h]
  __int64 v21; // [rsp-30h] [rbp-30h]
  __int64 v22; // [rsp-28h] [rbp-28h]
  __int64 v23; // [rsp-8h] [rbp-8h]

  v6 = *(unsigned int *)(a1 + 8);
  v7 = *a2;
  v8 = *(unsigned int *)(a1 + 12);
  v9 = *(_QWORD *)a1;
  v10 = *a3;
  v11 = *a4;
  v12 = *(_QWORD *)a1 + 24 * v6;
  if ( v6 >= v8 )
  {
    v23 = v4;
    v15 = (const __m128i *)&v19;
    v22 = v11;
    v16 = v6 + 1;
    v19 = v7;
    v20 = 0;
    v21 = v10;
    if ( v8 < v6 + 1 )
    {
      v17 = a1 + 16;
      if ( v9 > (unsigned __int64)&v19 || v12 <= (unsigned __int64)&v19 )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v16, 0x18u, v17, v16);
        v9 = *(_QWORD *)a1;
        v6 = *(unsigned int *)(a1 + 8);
      }
      else
      {
        v18 = (char *)&v19 - v9;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v16, 0x18u, v17, v16);
        v9 = *(_QWORD *)a1;
        v6 = *(unsigned int *)(a1 + 8);
        v15 = (const __m128i *)&v18[*(_QWORD *)a1];
      }
    }
    result = v9 + 24 * v6;
    *(__m128i *)result = _mm_loadu_si128(v15);
    *(_QWORD *)(result + 16) = v15[1].m128i_i64[0];
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    v13 = *(_DWORD *)(a1 + 8);
    if ( v12 )
    {
      *(_DWORD *)v12 = v7;
      *(_BYTE *)(v12 + 4) = 0;
      *(_QWORD *)(v12 + 8) = v10;
      *(_QWORD *)(v12 + 16) = v11;
      v13 = *(_DWORD *)(a1 + 8);
    }
    result = (unsigned int)(v13 + 1);
    *(_DWORD *)(a1 + 8) = result;
  }
  return result;
}
