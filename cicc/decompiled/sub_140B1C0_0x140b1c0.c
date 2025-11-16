// Function: sub_140B1C0
// Address: 0x140b1c0
//
__int64 __fastcall sub_140B1C0(__int64 a1, _QWORD *a2, char a3)
{
  __int64 v4; // rsi
  __int64 result; // rax
  char v6; // [rsp+Fh] [rbp-31h] BYREF
  __m128i v7; // [rsp+10h] [rbp-30h] BYREF
  unsigned __int8 v8; // [rsp+20h] [rbp-20h]

  v4 = sub_140ABA0(a1, a3, &v6);
  result = 0;
  if ( v4 )
  {
    if ( !v6 )
    {
      sub_140A980(&v7, v4, 0x17u, a2);
      return v8;
    }
  }
  return result;
}
