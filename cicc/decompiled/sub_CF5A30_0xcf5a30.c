// Function: sub_CF5A30
// Address: 0xcf5a30
//
__int64 __fastcall sub_CF5A30(_QWORD *a1, unsigned __int8 *a2, unsigned __int8 *a3, __int64 a4)
{
  int v6; // eax
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  __m128i v9; // xmm1
  __m128i v10; // xmm2
  __int64 result; // rax
  __int64 v12; // rdi
  _OWORD v14[3]; // [rsp+10h] [rbp-80h] BYREF
  __m128i v15; // [rsp+40h] [rbp-50h] BYREF
  __m128i v16; // [rsp+50h] [rbp-40h] BYREF
  __m128i v17[3]; // [rsp+60h] [rbp-30h] BYREF

  v6 = *a2;
  if ( (unsigned __int8)(v6 - 34) <= 0x33u )
  {
    v12 = 0x8000000000041LL;
    if ( _bittest64(&v12, (unsigned int)(v6 - 34)) )
      return sub_CF5550(a1, a2, a3, a4);
    v7 = (unsigned int)(v6 - 29);
  }
  else
  {
    v7 = (unsigned int)(v6 - 29);
    if ( (unsigned int)v7 > 0x38 )
      goto LABEL_4;
  }
  v8 = 0x110000800000220LL;
  if ( _bittest64(&v8, v7) )
    return 3;
LABEL_4:
  sub_D66840(&v15);
  v9 = _mm_loadu_si128(&v16);
  v10 = _mm_loadu_si128(v17);
  v14[0] = _mm_loadu_si128(&v15);
  v14[1] = v9;
  v14[2] = v10;
  result = sub_CF52B0(a1, a3, (__int64)v14, a4);
  if ( (_BYTE)result )
    return 3;
  return result;
}
