// Function: sub_2CB2D10
// Address: 0x2cb2d10
//
__int64 __fastcall sub_2CB2D10(__int64 a1)
{
  __int64 result; // rax
  char v2; // dl
  int v3; // eax

  result = 0;
  if ( *(_BYTE *)a1 == 85 )
  {
    result = *(_QWORD *)(a1 - 32);
    if ( result )
    {
      if ( !*(_BYTE *)result
        && *(_QWORD *)(result + 24) == *(_QWORD *)(a1 + 80)
        && (*(_BYTE *)(result + 33) & 0x20) != 0
        && (v2 = *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL), v2 != 3)
        && v2 )
      {
        v3 = *(_DWORD *)(result + 36);
        if ( v3 != 173 )
        {
          switch ( v3 )
          {
            case 8654:
            case 8656:
            case 8663:
            case 8669:
            case 8692:
            case 8694:
            case 8699:
            case 8701:
              return a1;
            default:
              return 0;
          }
        }
        return a1;
      }
      else
      {
        return 0;
      }
    }
  }
  return result;
}
