// Function: sub_E22E40
// Address: 0xe22e40
//
__int64 __fastcall sub_E22E40(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  char *v3; // rcx
  char v4; // dl

  result = *a2;
  if ( *a2 )
  {
    v3 = (char *)a2[1];
    v4 = *v3;
    *a2 = result - 1;
    a2[1] = (__int64)(v3 + 1);
    switch ( v4 )
    {
      case 'A':
        goto LABEL_5;
      case 'B':
        result = 1;
        break;
      case 'C':
        result = 2;
        break;
      case 'D':
        result = 3;
        break;
      case 'Q':
        result = 256;
        break;
      case 'R':
        result = 257;
        break;
      case 'S':
        result = 258;
        break;
      case 'T':
        result = 259;
        break;
      default:
        *(_BYTE *)(a1 + 8) = 1;
LABEL_5:
        result = 0;
        break;
    }
  }
  else
  {
    *(_BYTE *)(a1 + 8) = 1;
  }
  return result;
}
