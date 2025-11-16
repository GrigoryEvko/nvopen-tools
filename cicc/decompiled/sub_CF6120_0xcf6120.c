// Function: sub_CF6120
// Address: 0xcf6120
//
__int64 __fastcall sub_CF6120(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  unsigned int v4; // r15d
  _BYTE v8[96]; // [rsp+10h] [rbp-60h] BYREF

  v4 = 3;
  if ( !byte_3F70480[8 * ((*(_WORD *)(a2 + 2) >> 7) & 7) + 1] )
  {
    v4 = 2;
    if ( *a3 )
    {
      sub_D66630(v8);
      if ( !(unsigned __int8)sub_CF4D50(a1, (__int64)v8, (__int64)a3, a4, a2)
        || (sub_CF5020(a1, (__int64)a3, 0) & 2) == 0 )
      {
        return 0;
      }
    }
  }
  return v4;
}
