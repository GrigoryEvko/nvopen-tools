// Function: sub_BCAE30
// Address: 0xbcae30
//
__int64 __fastcall sub_BCAE30(__int64 a1)
{
  __int64 v2; // r12
  __int64 v3; // [rsp+20h] [rbp-20h]

  switch ( *(_BYTE *)(a1 + 8) )
  {
    case 0:
    case 1:
      v3 = 16;
      break;
    case 2:
      v3 = 32;
      break;
    case 3:
      v3 = 64;
      break;
    case 4:
      v3 = 80;
      break;
    case 5:
    case 6:
      v3 = 128;
      break;
    case 0xA:
      v3 = 0x2000;
      break;
    case 0xC:
      v3 = *(_DWORD *)(a1 + 8) >> 8;
      break;
    case 0x11:
    case 0x12:
      v2 = *(unsigned int *)(a1 + 32);
      v3 = v2 * sub_BCAE30(*(_QWORD *)(a1 + 24));
      break;
    default:
      v3 = 0;
      break;
  }
  return v3;
}
