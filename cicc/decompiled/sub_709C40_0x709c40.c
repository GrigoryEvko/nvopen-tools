// Function: sub_709C40
// Address: 0x709c40
//
__int64 __fastcall sub_709C40(const __m128i *a1, unsigned __int8 a2)
{
  __int64 v2; // rdx
  _BYTE *v4; // rax
  __int64 v5; // [rsp+0h] [rbp-10h] BYREF
  __int64 v6; // [rsp+8h] [rbp-8h] BYREF

  v5 = sub_709B30(a2, a1);
  v6 = v2;
  if ( unk_4F07580 )
  {
    if ( (HIWORD(v6) & 0x7FFF) != 0x7FFF )
      return 0;
  }
  else if ( (__ROL2__(v5, 8) & 0x7FFF) != 0x7FFF )
  {
    return 0;
  }
  v4 = (char *)&v5 + 2;
  if ( !BYTE2(v5) )
  {
    while ( (char *)&v6 + 6 != ++v4 )
    {
      if ( *v4 )
        return 1;
    }
    return 0;
  }
  return 1;
}
