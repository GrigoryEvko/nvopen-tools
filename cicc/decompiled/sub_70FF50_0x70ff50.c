// Function: sub_70FF50
// Address: 0x70ff50
//
__int64 __fastcall sub_70FF50(const __m128i *a1, __int64 a2, int a3, int a4, _DWORD *a5, unsigned __int8 *a6)
{
  __int64 result; // rax
  unsigned __int8 v10; // al
  unsigned __int8 v12; // [rsp+17h] [rbp-49h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-48h] BYREF
  int v14; // [rsp+1Ch] [rbp-44h] BYREF
  __int16 v15[32]; // [rsp+20h] [rbp-40h] BYREF

  sub_724A80(a2, 1);
  *(__m128i *)(a2 + 176) = _mm_loadu_si128(a1);
  sub_70FEF0(a2, &v12, &v13, &v14);
  result = sub_621140(a2, a2, v12);
  if ( !(_DWORD)result )
  {
    if ( a3 && !*a5 )
    {
      *a5 = 61;
      v10 = 5;
      if ( dword_4D04964 )
        v10 = byte_4F07472[0];
      *a6 = v10;
    }
    if ( a4 )
    {
      if ( (int)sub_6210B0(a2, 0) < 0 )
        *(__m128i *)(a2 + 176) = _mm_loadu_si128((const __m128i *)(16LL * v12 + 82864032));
      else
        *(__m128i *)(a2 + 176) = _mm_loadu_si128((const __m128i *)(16LL * v12 + 82863808));
    }
    else
    {
      sub_621EE0(v15, v14);
      sub_6213D0(a2 + 176, (__int64)v15);
    }
    result = v13;
    if ( v13 )
      return sub_6215A0((__int16 *)(a2 + 176), v14);
  }
  return result;
}
