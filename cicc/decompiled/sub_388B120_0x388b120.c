// Function: sub_388B120
// Address: 0x388b120
//
__int64 __fastcall sub_388B120(__int64 a1)
{
  __int64 v1; // r14
  unsigned int v2; // r13d
  unsigned __int64 v4[2]; // [rsp+0h] [rbp-50h] BYREF
  _BYTE v5[64]; // [rsp+10h] [rbp-40h] BYREF

  v1 = a1 + 8;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 3, "expected '=' after deplibs") )
    return 1;
  v2 = sub_388AF10(a1, 6, "expected '=' after deplibs");
  if ( (_BYTE)v2 )
  {
    return 1;
  }
  else if ( *(_DWORD *)(a1 + 64) == 7 )
  {
    *(_DWORD *)(a1 + 64) = sub_3887100(v1);
  }
  else
  {
    while ( 1 )
    {
      v4[0] = (unsigned __int64)v5;
      v4[1] = 0;
      v5[0] = 0;
      v2 = sub_388B0A0(a1, v4);
      if ( (_BYTE)v2 )
        break;
      if ( (_BYTE *)v4[0] != v5 )
        j_j___libc_free_0(v4[0]);
      if ( *(_DWORD *)(a1 + 64) != 4 )
        return sub_388AF10(a1, 7, "expected ']' at end of list");
      *(_DWORD *)(a1 + 64) = sub_3887100(v1);
    }
    if ( (_BYTE *)v4[0] != v5 )
      j_j___libc_free_0(v4[0]);
  }
  return v2;
}
