// Function: sub_FC78A0
// Address: 0xfc78a0
//
__int64 __fastcall sub_FC78A0(__int64 *a1, __int64 a2, __int64 a3, int a4, const void *a5, __int64 a6, int a7)
{
  const __m128i *v8; // r12
  __int64 v9; // rbx
  unsigned __int64 v10; // rdx
  int v11; // eax
  unsigned __int64 v12; // rcx
  __int64 v13; // rax
  unsigned __int64 v14; // r8
  size_t v15; // r14
  __m128i *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r12
  __int64 result; // rax
  unsigned __int64 v20; // rcx
  __int64 v21; // rdi
  const void *v22; // rsi
  char *v23; // r12
  __int64 v24; // [rsp+8h] [rbp-48h]
  _DWORD v25[2]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v26; // [rsp+18h] [rbp-38h]
  __int64 v27; // [rsp+20h] [rbp-30h]

  v8 = (const __m128i *)v25;
  v9 = *a1;
  v27 = a3;
  v26 = a2;
  v25[1] = a6;
  v10 = *(_QWORD *)(v9 + 72);
  v11 = (a4 << 31) | (4 * a7) & 0x7FFFFFFC | 1;
  v12 = *(unsigned int *)(v9 + 84);
  v25[0] = v11;
  v13 = *(unsigned int *)(v9 + 80);
  v14 = v13 + 1;
  if ( v13 + 1 > v12 )
  {
    v21 = v9 + 72;
    v22 = (const void *)(v9 + 88);
    if ( v10 > (unsigned __int64)v25 )
    {
      v24 = a6;
    }
    else
    {
      v24 = a6;
      if ( (unsigned __int64)v25 < v10 + 24 * v13 )
      {
        v23 = (char *)v25 - v10;
        sub_C8D5F0(v21, v22, v14, 0x18u, v14, a6);
        v10 = *(_QWORD *)(v9 + 72);
        v13 = *(unsigned int *)(v9 + 80);
        a6 = v24;
        v8 = (const __m128i *)&v23[v10];
        goto LABEL_2;
      }
    }
    sub_C8D5F0(v21, v22, v14, 0x18u, v14, a6);
    v10 = *(_QWORD *)(v9 + 72);
    v13 = *(unsigned int *)(v9 + 80);
    a6 = v24;
  }
LABEL_2:
  v15 = 8 * a6;
  v16 = (__m128i *)(v10 + 24 * v13);
  *v16 = _mm_loadu_si128(v8);
  v17 = v8[1].m128i_i64[0];
  v18 = (8 * a6) >> 3;
  v16[1].m128i_i64[0] = v17;
  result = *(unsigned int *)(v9 + 224);
  v20 = *(unsigned int *)(v9 + 228);
  ++*(_DWORD *)(v9 + 80);
  if ( v18 + result > v20 )
  {
    sub_C8D5F0(v9 + 216, (const void *)(v9 + 232), v18 + result, 8u, v14, a6);
    result = *(unsigned int *)(v9 + 224);
  }
  if ( v15 )
  {
    memcpy((void *)(*(_QWORD *)(v9 + 216) + 8 * result), a5, v15);
    result = *(unsigned int *)(v9 + 224);
  }
  *(_DWORD *)(v9 + 224) = result + v18;
  return result;
}
