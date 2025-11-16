// Function: sub_3847390
// Address: 0x3847390
//
void __fastcall sub_3847390(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        int a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v12; // rcx
  __int64 v13; // rsi
  int v14; // eax
  unsigned int v15; // ebx
  int v16; // r9d
  int v17; // r9d
  __m128i v18; // xmm0
  unsigned __int8 *v19; // rax
  __int64 v20; // r9
  unsigned __int8 *v21; // rdx
  unsigned __int8 *v22; // r15
  __int64 v23; // rdx
  unsigned __int8 *v24; // r14
  unsigned __int8 **v25; // rdx
  __int64 v26; // [rsp+20h] [rbp-60h] BYREF
  int v27; // [rsp+28h] [rbp-58h]
  __m128i v28; // [rsp+30h] [rbp-50h] BYREF
  __int128 v29; // [rsp+40h] [rbp-40h] BYREF

  v12 = a2;
  v13 = *(_QWORD *)(a2 + 80);
  v26 = v13;
  if ( v13 )
  {
    sub_B96E90((__int64)&v26, v13, 1);
    v12 = a2;
  }
  v14 = *(_DWORD *)(v12 + 72);
  v28 = 0;
  v27 = v14;
  v29 = 0;
  if ( a4 <= 1 )
  {
    v19 = sub_33FAF80(a1[1], 234, (__int64)&v26, a7, a8, a6, (__m128i)0LL);
    v22 = v21;
    v23 = *(unsigned int *)(a5 + 8);
    v24 = v19;
    if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
    {
      sub_C8D5F0(a5, (const void *)(a5 + 16), v23 + 1, 0x10u, v23 + 1, v20);
      v23 = *(unsigned int *)(a5 + 8);
    }
    v25 = (unsigned __int8 **)(*(_QWORD *)a5 + 16 * v23);
    *v25 = v24;
    v25[1] = v22;
    ++*(_DWORD *)(a5 + 8);
  }
  else
  {
    v15 = a4 >> 1;
    sub_375BC20(a1, a2, a3, (__int64)&v28, (__int64)&v29, (__m128i)0LL);
    if ( *(_BYTE *)sub_2E79000(*(__int64 **)(a1[1] + 40)) )
    {
      v18 = _mm_load_si128(&v28);
      v28.m128i_i64[0] = v29;
      v28.m128i_i32[2] = DWORD2(v29);
      *(_QWORD *)&v29 = v18.m128i_i64[0];
      DWORD2(v29) = v18.m128i_i32[2];
    }
    sub_3847390((_DWORD)a1, v28.m128i_i32[0], v28.m128i_i32[2], v15, a5, v16, a7, a8);
    sub_3847390((_DWORD)a1, v29, DWORD2(v29), v15, a5, v17, a7, a8);
  }
  if ( v26 )
    sub_B91220((__int64)&v26, v26);
}
