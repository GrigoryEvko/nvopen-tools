// Function: sub_1F59410
// Address: 0x1f59410
//
__int64 __fastcall sub_1F59410(__int64 a1)
{
  __int64 result; // rax
  unsigned int v2; // edx
  __int64 v3; // rbx
  char v4; // al

  switch ( *(_BYTE *)(a1 + 8) )
  {
    case 0:
      result = 112;
      break;
    case 1:
      result = 8;
      break;
    case 2:
      result = 9;
      break;
    case 3:
      result = 10;
      break;
    case 4:
      result = 11;
      break;
    case 5:
      result = 12;
      break;
    case 6:
      result = 13;
      break;
    case 9:
      result = 110;
      break;
    case 0xB:
      v2 = *(_DWORD *)(a1 + 8) >> 8;
      if ( v2 == 32 )
      {
        result = 5;
      }
      else if ( v2 > 0x20 )
      {
        result = 6;
        if ( v2 != 64 )
        {
          result = 0;
          if ( v2 == 128 )
            result = 7;
        }
      }
      else
      {
        result = 3;
        if ( v2 != 8 )
        {
          result = 4;
          if ( v2 != 16 )
          {
            LOBYTE(result) = v2 == 1;
            result = (unsigned int)(2 * result);
          }
        }
      }
      break;
    case 0xF:
      result = 4294967294LL;
      break;
    case 0x10:
      v3 = *(_QWORD *)(a1 + 32);
      v4 = sub_1F59410(*(_QWORD *)(a1 + 24), 0);
      result = sub_1D15020(v4, v3);
      break;
    default:
      result = 1;
      break;
  }
  return result;
}
