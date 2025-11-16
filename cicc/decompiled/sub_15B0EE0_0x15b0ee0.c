// Function: sub_15B0EE0
// Address: 0x15b0ee0
//
__int64 __fastcall sub_15B0EE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  int v4; // edx

  result = a1;
  switch ( a3 )
  {
    case 7LL:
      if ( *(_DWORD *)a2 == 1698983758 && *(_WORD *)(a2 + 4) == 30050 )
      {
        v4 = 0;
        if ( *(_BYTE *)(a2 + 6) == 103 )
          goto LABEL_10;
      }
      goto LABEL_6;
    case 9LL:
      if ( *(_QWORD *)a2 == 0x756265446C6C7546LL )
      {
        v4 = 1;
        if ( *(_BYTE *)(a2 + 8) == 103 )
          goto LABEL_10;
      }
      goto LABEL_6;
    case 14LL:
      if ( *(_QWORD *)a2 == 0x6C626154656E694CLL && *(_DWORD *)(a2 + 8) == 1850700645 )
      {
        v4 = 2;
        if ( *(_WORD *)(a2 + 12) == 31084 )
          goto LABEL_10;
      }
LABEL_6:
      *(_BYTE *)(a1 + 4) = 0;
      return result;
  }
  if ( a3 != 19
    || *(_QWORD *)a2 ^ 0x7269446775626544LL | *(_QWORD *)(a2 + 8) ^ 0x4F73657669746365LL
    || *(_WORD *)(a2 + 16) != 27758
    || *(_BYTE *)(a2 + 18) != 121 )
  {
    goto LABEL_6;
  }
  v4 = 3;
LABEL_10:
  *(_BYTE *)(a1 + 4) = 1;
  *(_DWORD *)a1 = v4;
  return result;
}
