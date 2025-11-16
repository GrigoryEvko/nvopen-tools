// Function: sub_AF3310
// Address: 0xaf3310
//
__int64 __fastcall sub_AF3310(__int64 a1, __int64 a2, int a3)
{
  int v4; // edx
  __int64 v5; // [rsp+0h] [rbp-8h]

  switch ( a2 )
  {
    case 7LL:
      if ( *(_DWORD *)a1 == 1634100548 && *(_WORD *)(a1 + 4) == 27765 && *(_BYTE *)(a1 + 6) == 116 )
      {
        v4 = 0;
        break;
      }
LABEL_3:
      LODWORD(v5) = a3;
      BYTE4(v5) = 0;
      return v5;
    case 3LL:
      if ( *(_WORD *)a1 != 20039 || *(_BYTE *)(a1 + 2) != 85 )
        goto LABEL_3;
      v4 = 1;
      break;
    case 5LL:
      if ( *(_DWORD *)a1 != 1819308097 || *(_BYTE *)(a1 + 4) != 101 )
        goto LABEL_3;
      v4 = 3;
      break;
    default:
      if ( a2 != 4 || *(_DWORD *)a1 != 1701736270 )
        goto LABEL_3;
      v4 = 2;
      break;
  }
  LODWORD(v5) = v4;
  BYTE4(v5) = 1;
  return v5;
}
