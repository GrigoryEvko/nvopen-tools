// Function: sub_FD60A0
// Address: 0xfd60a0
//
__int64 __fastcall sub_FD60A0(__int64 a1, unsigned __int8 *a2, _QWORD **a3)
{
  unsigned int v3; // r13d
  __int64 v6; // rbx
  unsigned __int8 *v7; // r15
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // r8
  __int64 v12; // rbx
  const __m128i *v13; // rbx
  _QWORD *v14; // rdi
  __m128i v15; // xmm2
  __int64 v16; // [rsp+10h] [rbp-80h]
  const __m128i *v17; // [rsp+10h] [rbp-80h]
  char v18; // [rsp+18h] [rbp-78h]
  __m128i v19[3]; // [rsp+20h] [rbp-70h] BYREF
  char v20; // [rsp+50h] [rbp-40h]

  v18 = *(_BYTE *)(a1 + 67);
  v3 = v18 & 8;
  if ( (v18 & 8) != 0 )
    return 3;
  if ( (unsigned __int8)sub_B46420((__int64)a2) || (unsigned __int8)sub_B46490((__int64)a2) )
  {
    v6 = *(_QWORD *)(a1 + 40);
    v16 = *(_QWORD *)(a1 + 48);
    if ( v6 == v16 )
    {
LABEL_13:
      v11 = *(_QWORD *)(a1 + 24);
      v12 = 48LL * *(unsigned int *)(a1 + 32);
      v17 = (const __m128i *)(v11 + v12);
      if ( v11 == v11 + v12 )
        return v3;
      v13 = *(const __m128i **)(a1 + 24);
      while ( 1 )
      {
        v14 = *a3;
        v19[0] = _mm_loadu_si128(v13);
        v19[1] = _mm_loadu_si128(v13 + 1);
        v15 = _mm_loadu_si128(v13 + 2);
        v20 = 1;
        v19[2] = v15;
        v3 |= sub_CF63E0(v14, a2, v19, (__int64)(a3 + 1));
        if ( (_BYTE)v3 == 3 )
          break;
        v13 += 3;
        if ( v17 == v13 )
          return v3;
      }
    }
    else
    {
      while ( 1 )
      {
        v7 = *(unsigned __int8 **)(v6 + 16);
        if ( (unsigned __int8)(*v7 - 34) > 0x33u )
          break;
        v9 = 0x8000000000041LL;
        if ( !_bittest64(&v9, (unsigned int)*v7 - 34) )
          break;
        if ( (unsigned __int8)(*a2 - 34) > 0x33u )
          break;
        v10 = 0x8000000000041LL;
        if ( !_bittest64(&v10, (unsigned int)*a2 - 34)
          || (unsigned __int8)sub_CF5A30(*a3, *(unsigned __int8 **)(v6 + 16), a2, (__int64)(a3 + 1))
          || (unsigned __int8)sub_CF5A30(*a3, a2, v7, (__int64)(a3 + 1)) )
        {
          break;
        }
        v6 += 24;
        if ( v16 == v6 )
          goto LABEL_13;
      }
    }
    return 3;
  }
  return v3;
}
