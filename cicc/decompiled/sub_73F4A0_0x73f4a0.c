// Function: sub_73F4A0
// Address: 0x73f4a0
//
__int64 __fastcall sub_73F4A0(__int64 a1, __int64 a2, int a3, unsigned int a4, int a5)
{
  const __m128i *v8; // rdi
  __int64 result; // rax

  if ( (*(_BYTE *)(a2 + 32) & 0x10) != 0 && a1 && *(_QWORD *)(*(_QWORD *)a1 + 96LL) )
    sub_895AD0();
  v8 = *(const __m128i **)(a2 + 40);
  if ( v8 )
  {
    if ( v8[1].m128i_i8[8] == 10 )
      v8 = (const __m128i *)v8[3].m128i_i64[1];
    if ( a4 )
      a4 = a3 == 0 ? 16 : 20;
    if ( !a5 )
      a4 = 128;
    result = (__int64)sub_73B8B0(v8, a4);
    if ( dword_4F077BC )
    {
      if ( (unsigned __int64)(qword_4F077A8 - 30400LL) <= 0x257F )
        result = sub_695350(result, a2, a5);
    }
    *(_BYTE *)(result + 25) |= 0x10u;
  }
  else
  {
    result = (__int64)sub_7305B0();
    *(_BYTE *)(result + 25) |= 0x10u;
  }
  return result;
}
