// Function: sub_254E0F0
// Address: 0x254e0f0
//
__int64 __fastcall sub_254E0F0(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r11d
  const __m128i *v7; // rbx
  const __m128i *v8; // r13
  char v11; // r15
  unsigned __int8 v12; // al
  __int64 v13; // rax
  __m128i v14; // xmm0
  __m128i v16; // [rsp+0h] [rbp-80h] BYREF
  __int64 v17; // [rsp+10h] [rbp-70h]
  unsigned __int8 v18; // [rsp+1Bh] [rbp-65h]
  int v19; // [rsp+1Ch] [rbp-64h]
  __int64 v20; // [rsp+20h] [rbp-60h]
  __m128i *v21; // [rsp+28h] [rbp-58h]
  char v22; // [rsp+3Fh] [rbp-41h] BYREF
  char v23; // [rsp+40h] [rbp-40h] BYREF

  v6 = *(unsigned __int8 *)(a1 + 105);
  v20 = a2;
  if ( (_BYTE)v6 )
  {
    v7 = *(const __m128i **)(a1 + 144);
    v22 = 0;
    v19 = a4;
    v8 = (const __m128i *)((char *)v7 + 24 * *(unsigned int *)(a1 + 152));
    v21 = (__m128i *)&v23;
    if ( v7 != v8 )
    {
      v18 = v6;
      v11 = a5;
      v17 = a1;
      do
      {
        if ( (v7[1].m128i_i8[0] & a4) != 0 )
        {
          if ( !v11
            || (v12 = *(_BYTE *)v7->m128i_i64[0], v12 <= 0x1Cu)
            || (v12 & 0xFD) != 0x54
            || (sub_250D230((unsigned __int64 *)v21, v7->m128i_i64[0], 1, 0),
                !(unsigned __int8)sub_2526B50(v20, v21, v17, a3, v19, &v22, 1u)) )
          {
            v13 = *(unsigned int *)(a3 + 8);
            v14 = _mm_loadu_si128(v7);
            if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
            {
              v16 = v14;
              sub_C8D5F0(a3, (const void *)(a3 + 16), v13 + 1, 0x10u, a5, a6);
              v13 = *(unsigned int *)(a3 + 8);
              v14 = _mm_load_si128(&v16);
            }
            *(__m128i *)(*(_QWORD *)a3 + 16 * v13) = v14;
            ++*(_DWORD *)(a3 + 8);
          }
        }
        v7 = (const __m128i *)((char *)v7 + 24);
      }
      while ( v8 != v7 );
      return v18;
    }
  }
  return v6;
}
