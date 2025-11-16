// Function: sub_CEA320
// Address: 0xcea320
//
__int64 __fastcall sub_CEA320(unsigned int a1)
{
  __int64 result; // rax

  result = 0;
  if ( a1 <= 0x2903 )
  {
    if ( a1 > 0x28BE )
    {
      switch ( a1 )
      {
        case 0x28BFu:
        case 0x28C1u:
        case 0x28C3u:
        case 0x28C5u:
        case 0x28C7u:
        case 0x28C9u:
        case 0x28CBu:
        case 0x28CDu:
        case 0x28CFu:
        case 0x28D1u:
        case 0x28D3u:
        case 0x28D5u:
        case 0x28D7u:
        case 0x28D9u:
        case 0x28DBu:
        case 0x28DDu:
        case 0x28DFu:
        case 0x28E1u:
        case 0x28E3u:
        case 0x28E5u:
        case 0x28E7u:
        case 0x28E9u:
        case 0x28EBu:
        case 0x28EDu:
        case 0x28EFu:
        case 0x28F1u:
        case 0x28F3u:
        case 0x28F5u:
        case 0x28F7u:
        case 0x28F9u:
        case 0x28FBu:
        case 0x28FDu:
        case 0x28FFu:
        case 0x2901u:
        case 0x2903u:
LABEL_6:
          result = 1;
          break;
        default:
LABEL_7:
          result = 0;
          break;
      }
    }
    else if ( a1 > 0x2535 )
    {
      switch ( a1 )
      {
        case 0x264Bu:
        case 0x264Eu:
        case 0x264Fu:
        case 0x2654u:
        case 0x2657u:
        case 0x2659u:
        case 0x265Cu:
          goto LABEL_6;
        case 0x264Cu:
        case 0x264Du:
        case 0x2650u:
        case 0x2651u:
        case 0x2652u:
        case 0x2653u:
        case 0x2655u:
        case 0x2656u:
        case 0x2658u:
        case 0x265Au:
        case 0x265Bu:
          goto LABEL_7;
        default:
          return result;
      }
    }
    else if ( a1 > 0x2502 )
    {
      switch ( a1 )
      {
        case 0x2503u:
        case 0x2506u:
        case 0x2509u:
        case 0x250Bu:
        case 0x250Du:
        case 0x250Fu:
        case 0x2511u:
        case 0x2513u:
        case 0x2515u:
        case 0x2517u:
        case 0x2519u:
        case 0x251Bu:
        case 0x251Du:
        case 0x251Fu:
        case 0x2521u:
        case 0x2523u:
        case 0x2525u:
        case 0x2527u:
        case 0x2529u:
        case 0x252Bu:
        case 0x252Du:
        case 0x252Fu:
        case 0x2531u:
        case 0x2533u:
        case 0x2535u:
          goto LABEL_6;
        default:
          goto LABEL_7;
      }
    }
  }
  return result;
}
