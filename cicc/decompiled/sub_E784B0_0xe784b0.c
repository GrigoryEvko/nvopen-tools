// Function: sub_E784B0
// Address: 0xe784b0
//
__int64 __fastcall sub_E784B0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rdi
  __int64 v7; // rsi
  unsigned int v8; // ecx
  __int64 v9; // r9
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rdi
  __m128i v13; // xmm0
  __m128i *v14; // rsi
  unsigned int v15; // r10d
  __m128i v16; // [rsp+0h] [rbp-40h] BYREF
  __m128i v17; // [rsp+10h] [rbp-30h] BYREF
  __m128i v18; // [rsp+20h] [rbp-20h] BYREF

  result = *(_QWORD *)a2;
  if ( !*(_QWORD *)a2 )
  {
    if ( (*(_BYTE *)(a2 + 9) & 0x70) != 0x20 || *(char *)(a2 + 8) < 0 )
      BUG();
    *(_BYTE *)(a2 + 8) |= 8u;
    result = sub_E807D0(*(_QWORD *)(a2 + 24));
    *(_QWORD *)a2 = result;
  }
  v5 = *(unsigned int *)(a1 + 24);
  v6 = *(_QWORD *)(result + 8);
  v7 = *(_QWORD *)(a1 + 8);
  if ( (_DWORD)v5 )
  {
    v8 = (v5 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    result = v7 + 16LL * v8;
    v9 = *(_QWORD *)result;
    if ( v6 == *(_QWORD *)result )
    {
LABEL_4:
      if ( result != v7 + 16 * v5 )
      {
        v10 = *(unsigned int *)(result + 8);
        v11 = *(_QWORD *)(a1 + 32);
        result = 32LL * *(unsigned int *)(a1 + 40);
        v12 = v11 + 32 * v10;
        if ( v12 != result + v11 )
        {
          result = *(_QWORD *)(v12 + 16);
          v13 = _mm_loadu_si128((const __m128i *)(result - 48));
          v16 = v13;
          v17.m128i_i64[1] = _mm_loadu_si128((const __m128i *)(result - 32)).m128i_i64[1];
          v17.m128i_i64[0] = a2;
          v18 = _mm_loadu_si128((const __m128i *)(result - 16));
          v18.m128i_i8[8] = 1;
          v14 = *(__m128i **)(v12 + 16);
          if ( v14 == *(__m128i **)(v12 + 24) )
          {
            return sub_E782B0((const __m128i **)(v12 + 8), v14, &v16);
          }
          else
          {
            if ( v14 )
            {
              *v14 = v13;
              v14[1] = _mm_loadu_si128(&v17);
              v14[2] = _mm_loadu_si128(&v18);
              v14 = *(__m128i **)(v12 + 16);
            }
            *(_QWORD *)(v12 + 16) = v14 + 3;
          }
        }
      }
    }
    else
    {
      result = 1;
      while ( v9 != -4096 )
      {
        v15 = result + 1;
        v8 = (v5 - 1) & (result + v8);
        result = v7 + 16LL * v8;
        v9 = *(_QWORD *)result;
        if ( v6 == *(_QWORD *)result )
          goto LABEL_4;
        result = v15;
      }
    }
  }
  return result;
}
