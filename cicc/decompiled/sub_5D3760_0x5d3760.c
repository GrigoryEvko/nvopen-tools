// Function: sub_5D3760
// Address: 0x5d3760
//
__m128i __fastcall sub_5D3760(__int64 a1)
{
  __int64 i; // rax
  __int64 v2; // rdx
  _QWORD v4[2]; // [rsp+0h] [rbp-10h] BYREF

  for ( i = *(_QWORD *)(a1 + 128); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v4[0] = sub_709B30(*(unsigned __int8 *)(i + 160), a1 + 176);
  v4[1] = v2;
  return _mm_cvtsi32_si128(sub_12F9960(v4));
}
