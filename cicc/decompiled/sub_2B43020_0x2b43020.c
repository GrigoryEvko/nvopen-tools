// Function: sub_2B43020
// Address: 0x2b43020
//
__int64 __fastcall sub_2B43020(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // rcx
  void *v10; // rdi
  __m128i v11; // xmm0
  int v12; // eax
  unsigned int v13; // r13d
  __int64 v14; // r13
  unsigned int v15; // r15d
  __int64 result; // rax
  __int64 v17; // rdx
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // r14
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned __int64 v27; // rdi
  int v28; // eax
  size_t v29; // rdx
  __int64 v30; // [rsp+8h] [rbp-48h]
  int v31; // [rsp+8h] [rbp-48h]
  unsigned __int64 v32[7]; // [rsp+18h] [rbp-38h] BYREF

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x800000000LL;
  if ( *(_DWORD *)(a2 + 8) )
    sub_2B0CFB0(a1, a2, a3, a4, a5, a6);
  *(_QWORD *)(a1 + 80) = 6;
  v8 = *(_QWORD *)(a2 + 96);
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = v8;
  LOBYTE(a4) = v8 != 0;
  LOBYTE(a3) = v8 != -4096;
  if ( ((unsigned __int8)a3 & (v8 != 0)) != 0 && v8 != -8192 )
    sub_BD6050((unsigned __int64 *)(a1 + 80), *(_QWORD *)(a2 + 80) & 0xFFFFFFFFFFFFFFF8LL);
  *(_QWORD *)(a1 + 104) = *(_QWORD *)(a2 + 104);
  *(_QWORD *)(a1 + 112) = a1 + 128;
  *(_QWORD *)(a1 + 120) = 0x400000000LL;
  if ( *(_DWORD *)(a2 + 120) )
    sub_2B0D670(a1 + 112, a2 + 112, a3, a4, a5, a6);
  *(_QWORD *)(a1 + 144) = a1 + 160;
  *(_QWORD *)(a1 + 152) = 0x400000000LL;
  v9 = *(unsigned int *)(a2 + 152);
  if ( (_DWORD)v9 )
    sub_2B0D430(a1 + 144, a2 + 144, a3, v9, a5, a6);
  v10 = (void *)(a1 + 224);
  v11 = _mm_loadu_si128((const __m128i *)(a2 + 184));
  *(_QWORD *)(a1 + 176) = *(_QWORD *)(a2 + 176);
  v12 = *(_DWORD *)(a2 + 200);
  *(__m128i *)(a1 + 184) = v11;
  *(_DWORD *)(a1 + 200) = v12;
  *(_QWORD *)(a1 + 208) = a1 + 224;
  *(_QWORD *)(a1 + 216) = 0x200000000LL;
  v13 = *(_DWORD *)(a2 + 216);
  if ( v13 )
  {
    a5 = a1 + 208;
    if ( a1 + 208 != a2 + 208 )
    {
      v29 = 8LL * v13;
      if ( v13 <= 2
        || (sub_C8D5F0(a1 + 208, (const void *)(a1 + 224), v13, 8u, a5, v13),
            v10 = *(void **)(a1 + 208),
            (v29 = 8LL * *(unsigned int *)(a2 + 216)) != 0) )
      {
        memcpy(v10, *(const void **)(a2 + 208), v29);
      }
      *(_DWORD *)(a1 + 216) = v13;
    }
  }
  v14 = a1 + 256;
  *(_QWORD *)(a1 + 240) = a1 + 256;
  *(_QWORD *)(a1 + 248) = 0x200000000LL;
  v15 = *(_DWORD *)(a2 + 248);
  if ( v15 && a1 + 240 != a2 + 240 )
  {
    v17 = v15;
    if ( v15 > 2 )
    {
      v22 = sub_C8D7D0(a1 + 240, a1 + 256, v15, 0x50u, v32, a6);
      sub_2B42CC0(a1 + 240, v22, v23, v24, v25, v26);
      v27 = *(_QWORD *)(a1 + 240);
      v28 = v32[0];
      if ( v14 != v27 )
      {
        v31 = v32[0];
        _libc_free(v27);
        v28 = v31;
      }
      *(_QWORD *)(a1 + 240) = v22;
      v17 = *(unsigned int *)(a2 + 248);
      v14 = v22;
      *(_DWORD *)(a1 + 252) = v28;
    }
    v18 = *(_QWORD *)(a2 + 240);
    v19 = v18 + 80 * v17;
    if ( v18 != v19 )
    {
      do
      {
        while ( 1 )
        {
          if ( v14 )
          {
            *(_DWORD *)(v14 + 8) = 0;
            *(_QWORD *)v14 = v14 + 16;
            *(_DWORD *)(v14 + 12) = 8;
            v20 = *(unsigned int *)(v18 + 8);
            if ( (_DWORD)v20 )
              break;
          }
          v18 += 80;
          v14 += 80;
          if ( v19 == v18 )
            goto LABEL_21;
        }
        v21 = v18;
        v30 = v19;
        v18 += 80;
        sub_2B0CFB0(v14, v21, v20, v9, a5, a6);
        v19 = v30;
        v14 += 80;
      }
      while ( v30 != v18 );
    }
LABEL_21:
    *(_DWORD *)(a1 + 248) = v15;
  }
  result = *(unsigned int *)(a2 + 432);
  *(__m128i *)(a1 + 416) = _mm_loadu_si128((const __m128i *)(a2 + 416));
  *(_DWORD *)(a1 + 432) = result;
  return result;
}
