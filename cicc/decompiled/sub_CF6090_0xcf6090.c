// Function: sub_CF6090
// Address: 0xcf6090
//
__int64 __fastcall sub_CF6090(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  unsigned int v4; // r13d
  unsigned int v7; // eax
  _BYTE v9[96]; // [rsp+10h] [rbp-60h] BYREF

  v4 = 3;
  if ( !byte_3F70480[8 * ((*(_WORD *)(a2 + 2) >> 7) & 7) + 1] )
  {
    v4 = 1;
    if ( *a3 )
    {
      sub_D665A0(v9);
      v7 = sub_CF4D50(a1, (__int64)v9, (__int64)a3, a4, a2);
      if ( !(_BYTE)v7 )
        return v7;
    }
  }
  return v4;
}
