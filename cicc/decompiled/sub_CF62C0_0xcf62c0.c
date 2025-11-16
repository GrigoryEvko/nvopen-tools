// Function: sub_CF62C0
// Address: 0xcf62c0
//
__int64 __fastcall sub_CF62C0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  unsigned int v4; // r15d
  unsigned int v7; // eax
  _BYTE v9[96]; // [rsp+10h] [rbp-60h] BYREF

  v4 = 3;
  if ( !byte_3F70480[8 * ((*(_WORD *)(a2 + 2) >> 2) & 7) + 2] )
  {
    if ( *a3 )
    {
      sub_D66720(v9);
      v7 = sub_CF4D50(a1, (__int64)v9, (__int64)a3, a4, a2);
      if ( !(_BYTE)v7 )
        return v7;
    }
  }
  return v4;
}
