// Function: sub_1B752E0
// Address: 0x1b752e0
//
void __fastcall sub_1B752E0(__int64 *a1, __int64 a2, __int64 a3, int a4, const void *a5, __int64 a6, int a7)
{
  __int64 v8; // rbx
  __int64 v9; // rax
  size_t v10; // r12
  __m128i *v11; // rax
  __int64 v12; // rdx
  unsigned __int64 v13; // r14
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // [rsp+8h] [rbp-48h]
  __m128i v17; // [rsp+10h] [rbp-40h] BYREF
  __int64 v18; // [rsp+20h] [rbp-30h]

  v8 = *a1;
  v17.m128i_i64[1] = a2;
  v18 = a3;
  v17.m128i_i32[1] = a6;
  v17.m128i_i32[0] = (a4 << 31) | (4 * a7) & 0x7FFFFFFC | 1;
  v9 = *(unsigned int *)(v8 + 80);
  if ( (unsigned int)v9 >= *(_DWORD *)(v8 + 84) )
  {
    v16 = a6;
    sub_16CD150(v8 + 72, (const void *)(v8 + 88), 0, 24, (int)a5, a6);
    v9 = *(unsigned int *)(v8 + 80);
    a6 = v16;
  }
  v10 = 8 * a6;
  v11 = (__m128i *)(*(_QWORD *)(v8 + 72) + 24 * v9);
  v12 = v18;
  v13 = (8 * a6) >> 3;
  *v11 = _mm_loadu_si128(&v17);
  v11[1].m128i_i64[0] = v12;
  v14 = *(unsigned int *)(v8 + 224);
  v15 = *(unsigned int *)(v8 + 228);
  ++*(_DWORD *)(v8 + 80);
  if ( v13 > v15 - v14 )
  {
    sub_16CD150(v8 + 216, (const void *)(v8 + 232), v13 + v14, 8, (int)a5, a6);
    v14 = *(unsigned int *)(v8 + 224);
  }
  if ( v10 )
  {
    memcpy((void *)(*(_QWORD *)(v8 + 216) + 8 * v14), a5, v10);
    LODWORD(v14) = *(_DWORD *)(v8 + 224);
  }
  *(_DWORD *)(v8 + 224) = v13 + v14;
}
