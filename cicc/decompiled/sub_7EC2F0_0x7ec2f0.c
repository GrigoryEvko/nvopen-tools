// Function: sub_7EC2F0
// Address: 0x7ec2f0
//
__int64 __fastcall sub_7EC2F0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  const __m128i *v3; // r12
  const __m128i *v4; // rax
  const __m128i *v5; // rax

  result = *(unsigned __int8 *)(a1 + 56);
  if ( (*(_BYTE *)(a1 + 56) & 0xFD) == 0x60 )
  {
    v3 = *(const __m128i **)(a1 + 72);
  }
  else
  {
    result = (unsigned int)(result - 106);
    if ( (result & 0xFD) != 0 )
      return result;
    result = *(_QWORD *)(a1 + 72);
    v3 = *(const __m128i **)(result + 16);
  }
  if ( v3 )
  {
    v4 = (const __m128i *)sub_730FF0(v3);
    v5 = sub_7EC2A0(v4, a2);
    return sub_730620((__int64)v3, v5);
  }
  return result;
}
