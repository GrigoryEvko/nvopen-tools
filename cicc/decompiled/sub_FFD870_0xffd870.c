// Function: sub_FFD870
// Address: 0xffd870
//
void __fastcall sub_FFD870(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // rbx
  __int64 v12; // rdx
  unsigned __int64 **v13; // rdi
  __m128i v14; // xmm0
  __int64 v15; // rdx
  __int64 *v16; // rsi
  _BYTE *v17; // rdi
  unsigned __int64 v18; // rax
  __m128i *v19; // rdx
  const __m128i *v20; // rbx
  unsigned __int64 v21; // r11
  unsigned __int64 v22; // rbx
  unsigned __int64 *v23; // rsi
  __m128i v24; // [rsp+0h] [rbp-520h] BYREF
  __int64 v25; // [rsp+10h] [rbp-510h]
  unsigned __int64 **v26; // [rsp+18h] [rbp-508h]
  unsigned __int64 *v27; // [rsp+20h] [rbp-500h] BYREF
  __int64 v28; // [rsp+28h] [rbp-4F8h]
  _BYTE v29[512]; // [rsp+30h] [rbp-4F0h] BYREF
  __int64 *v30; // [rsp+230h] [rbp-2F0h] BYREF
  __int64 v31; // [rsp+238h] [rbp-2E8h]
  _BYTE v32[736]; // [rsp+240h] [rbp-2E0h] BYREF

  v6 = *(_QWORD *)(a1 + 552);
  if ( *(_BYTE *)(a1 + 560) == 1 && v6 )
  {
    do
    {
      v8 = *(unsigned int *)(a1 + 8);
      v9 = *(_QWORD *)(a1 + 536);
      if ( v9 == v8 )
        return;
      v10 = *(_QWORD *)a1 + 32 * v8;
      v11 = *(_QWORD *)a1 + 32 * v9;
      if ( *(_BYTE *)v11 )
      {
        v30 = (__int64 *)v32;
        v31 = 0x200000000LL;
        if ( v11 == v10 )
        {
          v15 = 0;
          v16 = (__int64 *)v32;
        }
        else
        {
          v18 = v11 + 8;
          v16 = (__int64 *)v32;
          v15 = 0;
          while ( 1 )
          {
            v20 = (const __m128i *)v18;
            if ( !*(_BYTE *)(v18 - 8) )
              break;
            v21 = v15 + 1;
            if ( v15 + 1 > (unsigned __int64)HIDWORD(v31) )
            {
              if ( (unsigned __int64)v16 > v18 || (unsigned __int64)&v16[3 * v15] <= v18 )
              {
                v24.m128i_i64[0] = v18;
                v26 = (unsigned __int64 **)v10;
                sub_C8D5F0((__int64)&v30, v32, v21, 0x18u, v10, a6);
                v16 = v30;
                v15 = (unsigned int)v31;
                v10 = (__int64)v26;
                v18 = v24.m128i_i64[0];
              }
              else
              {
                v22 = v18 - (_QWORD)v16;
                v24.m128i_i64[0] = v10;
                v26 = (unsigned __int64 **)v18;
                sub_C8D5F0((__int64)&v30, v32, v21, 0x18u, v10, a6);
                v16 = v30;
                v15 = (unsigned int)v31;
                v10 = v24.m128i_i64[0];
                v18 = (unsigned __int64)v26;
                v20 = (const __m128i *)((char *)v30 + v22);
              }
            }
            v19 = (__m128i *)&v16[3 * v15];
            *v19 = _mm_loadu_si128(v20);
            v19[1].m128i_i64[0] = v20[1].m128i_i64[0];
            v16 = v30;
            v15 = (unsigned int)(v31 + 1);
            LODWORD(v31) = v31 + 1;
            if ( v10 == v18 + 24 )
              break;
            v18 += 32LL;
          }
        }
        sub_FFD380(a1, v16, v15);
        v17 = v30;
        *(_QWORD *)(a1 + 536) += (unsigned int)v31;
        if ( v17 == v32 )
          continue;
      }
      else
      {
        v12 = 0;
        v27 = (unsigned __int64 *)v29;
        v28 = 0x2000000000LL;
        if ( v11 == v10 )
        {
          v23 = (unsigned __int64 *)v29;
        }
        else
        {
          v13 = &v27;
          do
          {
            if ( *(_BYTE *)v11 )
              break;
            v14 = _mm_loadu_si128((const __m128i *)(v11 + 8));
            if ( v12 + 1 > (unsigned __int64)HIDWORD(v28) )
            {
              v25 = v10;
              v26 = v13;
              v24 = v14;
              sub_C8D5F0((__int64)v13, v29, v12 + 1, 0x10u, v10, a6);
              v12 = (unsigned int)v28;
              v10 = v25;
              v14 = _mm_load_si128(&v24);
              v13 = v26;
            }
            v11 += 32;
            *(__m128i *)&v27[2 * v12] = v14;
            v12 = (unsigned int)(v28 + 1);
            LODWORD(v28) = v28 + 1;
          }
          while ( v11 != v10 );
          v23 = v27;
        }
        sub_B26B80((__int64)&v30, v23, v12, 1u);
        v16 = (__int64 *)&v30;
        sub_B2A420(v6, (__int64)&v30, 0);
        sub_B1AA80((__int64)&v30, (__int64)&v30);
        v17 = v27;
        *(_QWORD *)(a1 + 536) += (unsigned int)v28;
        if ( v17 == v29 )
          continue;
      }
      _libc_free(v17, v16);
    }
    while ( *(_QWORD *)(a1 + 552) );
  }
}
