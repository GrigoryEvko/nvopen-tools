// Function: sub_AF3210
// Address: 0xaf3210
//
__int64 __fastcall sub_AF3210(__int64 a1, __int64 a2, int a3)
{
  int v4; // edx
  __int64 v5; // [rsp+0h] [rbp-8h]

  switch ( a2 )
  {
    case 7LL:
      if ( *(_DWORD *)a1 == 1698983758 && *(_WORD *)(a1 + 4) == 30050 && *(_BYTE *)(a1 + 6) == 103 )
      {
        v4 = 0;
        break;
      }
LABEL_3:
      LODWORD(v5) = a3;
      BYTE4(v5) = 0;
      return v5;
    case 9LL:
      if ( *(_QWORD *)a1 != 0x756265446C6C7546LL || *(_BYTE *)(a1 + 8) != 103 )
        goto LABEL_3;
      v4 = 1;
      break;
    case 14LL:
      if ( *(_QWORD *)a1 != 0x6C626154656E694CLL || *(_DWORD *)(a1 + 8) != 1850700645 || *(_WORD *)(a1 + 12) != 31084 )
        goto LABEL_3;
      v4 = 2;
      break;
    default:
      if ( a2 != 19
        || *(_QWORD *)a1 ^ 0x7269446775626544LL | *(_QWORD *)(a1 + 8) ^ 0x4F73657669746365LL
        || *(_WORD *)(a1 + 16) != 27758
        || *(_BYTE *)(a1 + 18) != 121 )
      {
        goto LABEL_3;
      }
      v4 = 3;
      break;
  }
  LODWORD(v5) = v4;
  BYTE4(v5) = 1;
  return v5;
}
