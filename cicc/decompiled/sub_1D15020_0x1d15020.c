// Function: sub_1D15020
// Address: 0x1d15020
//
__int64 __fastcall sub_1D15020(char a1, int a2)
{
  __int64 result; // rax

  switch ( a1 )
  {
    case 2:
      switch ( a2 )
      {
        case 1:
          result = 14;
          break;
        case 2:
          result = 15;
          break;
        case 4:
          result = 16;
          break;
        case 8:
          result = 17;
          break;
        case 16:
          result = 18;
          break;
        case 32:
          result = 19;
          break;
        case 64:
          result = 20;
          break;
        case 128:
          result = 21;
          break;
        case 512:
          result = 22;
          break;
        case 1024:
          result = 23;
          break;
        default:
          goto LABEL_3;
      }
      break;
    case 3:
      switch ( a2 )
      {
        case 1:
          result = 24;
          break;
        case 2:
          result = 25;
          break;
        case 4:
          result = 26;
          break;
        case 8:
          result = 27;
          break;
        case 16:
          result = 28;
          break;
        case 32:
          result = 29;
          break;
        case 64:
          result = 30;
          break;
        case 128:
          result = 31;
          break;
        case 256:
          result = 32;
          break;
        default:
          goto LABEL_3;
      }
      break;
    case 4:
      switch ( a2 )
      {
        case 1:
          result = 33;
          break;
        case 2:
          result = 34;
          break;
        case 4:
          result = 35;
          break;
        case 8:
          result = 36;
          break;
        case 16:
          result = 37;
          break;
        case 32:
          result = 38;
          break;
        case 64:
          result = 39;
          break;
        case 128:
          result = 40;
          break;
        default:
          goto LABEL_3;
      }
      break;
    case 5:
      switch ( a2 )
      {
        case 1:
          result = 41;
          break;
        case 2:
          result = 42;
          break;
        case 4:
          result = 43;
          break;
        case 8:
          result = 44;
          break;
        case 16:
          result = 45;
          break;
        case 32:
          result = 46;
          break;
        case 64:
          result = 47;
          break;
        case 128:
          result = 48;
          break;
        default:
          goto LABEL_3;
      }
      break;
    case 6:
      switch ( a2 )
      {
        case 1:
          result = 49;
          break;
        case 2:
          result = 50;
          break;
        case 4:
          result = 51;
          break;
        case 8:
          result = 52;
          break;
        case 16:
          result = 53;
          break;
        case 32:
          result = 54;
          break;
        default:
          goto LABEL_3;
      }
      break;
    case 7:
      if ( a2 != 1 )
        goto LABEL_3;
      result = 55;
      break;
    case 8:
      switch ( a2 )
      {
        case 2:
          result = 86;
          break;
        case 4:
          result = 87;
          break;
        case 8:
          result = 88;
          break;
        default:
          goto LABEL_3;
      }
      break;
    case 9:
      switch ( a2 )
      {
        case 1:
          result = 89;
          break;
        case 2:
          result = 90;
          break;
        case 4:
          result = 91;
          break;
        case 8:
          result = 92;
          break;
        case 16:
          result = 93;
          break;
        default:
          goto LABEL_3;
      }
      break;
    case 10:
      switch ( a2 )
      {
        case 1:
          result = 94;
          break;
        case 2:
          result = 95;
          break;
        case 4:
          result = 96;
          break;
        case 8:
          result = 97;
          break;
        default:
          goto LABEL_3;
      }
      break;
    default:
LABEL_3:
      result = 0;
      break;
  }
  return result;
}
