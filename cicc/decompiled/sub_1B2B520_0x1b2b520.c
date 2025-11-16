// Function: sub_1B2B520
// Address: 0x1b2b520
//
void __fastcall sub_1B2B520(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 i; // rbx
  _QWORD *v6; // rcx
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rsi
  int v12; // r8d
  unsigned int v13; // ecx
  __int64 *v14; // rdx
  __int64 v15; // r9
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __m128i v19; // xmm1
  __m128i v20; // xmm2
  __int64 v21; // rax
  _QWORD *v22; // rax
  unsigned __int8 v23; // dl
  int v24; // edx
  int v25; // r10d
  __m128i v26; // [rsp+0h] [rbp-50h] BYREF
  __m128i v27; // [rsp+10h] [rbp-40h] BYREF
  __m128i v28; // [rsp+20h] [rbp-30h] BYREF

  for ( i = *(_QWORD *)(a2 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    v22 = sub_1648700(i);
    v23 = *((_BYTE *)v22 + 16);
    if ( v23 > 0x17u )
    {
      v26.m128i_i64[0] = 0;
      v26.m128i_i32[2] = 1;
      v27 = 0u;
      v28.m128i_i64[0] = 0;
      v28.m128i_i8[8] = 0;
      if ( v23 == 77 )
      {
        v6 = (*((_BYTE *)v22 + 23) & 0x40) != 0 ? (_QWORD *)*(v22 - 1) : &v22[-3 * (*((_DWORD *)v22 + 5) & 0xFFFFFFF)];
        v7 = *((unsigned int *)v22 + 14);
        v26.m128i_i32[2] = 2;
        v8 = v6[3 * v7 + 1 + -1431655765 * (unsigned int)((i - (__int64)v6) >> 3)];
      }
      else
      {
        v8 = v22[5];
      }
      v9 = *(_QWORD *)(a1 + 24);
      v10 = *(unsigned int *)(v9 + 48);
      if ( (_DWORD)v10 )
      {
        v11 = *(_QWORD *)(v9 + 32);
        v12 = v10 - 1;
        v13 = (v10 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v14 = (__int64 *)(v11 + 16LL * v13);
        v15 = *v14;
        if ( v8 == *v14 )
        {
LABEL_8:
          if ( v14 != (__int64 *)(v11 + 16 * v10) )
          {
            v16 = v14[1];
            if ( v16 )
            {
              v17 = *(_QWORD *)(v16 + 48);
              v27.m128i_i64[1] = i;
              v26.m128i_i64[0] = v17;
              v18 = *(unsigned int *)(a3 + 8);
              if ( (unsigned int)v18 >= *(_DWORD *)(a3 + 12) )
              {
                sub_16CD150(a3, (const void *)(a3 + 16), 0, 48, v12, v15);
                v18 = *(unsigned int *)(a3 + 8);
              }
              v19 = _mm_loadu_si128(&v27);
              v20 = _mm_loadu_si128(&v28);
              v21 = *(_QWORD *)a3 + 48 * v18;
              *(__m128i *)v21 = _mm_loadu_si128(&v26);
              *(__m128i *)(v21 + 16) = v19;
              *(__m128i *)(v21 + 32) = v20;
              ++*(_DWORD *)(a3 + 8);
            }
          }
        }
        else
        {
          v24 = 1;
          while ( v15 != -8 )
          {
            v25 = v24 + 1;
            v13 = v12 & (v24 + v13);
            v14 = (__int64 *)(v11 + 16LL * v13);
            v15 = *v14;
            if ( v8 == *v14 )
              goto LABEL_8;
            v24 = v25;
          }
        }
      }
    }
  }
}
