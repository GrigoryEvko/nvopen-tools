// Function: sub_9208B0
// Address: 0x9208b0
//
__int64 __fastcall sub_9208B0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v5; // rsi
  __int64 v6; // r13
  char v7; // r12

  while ( 1 )
  {
    switch ( *(_BYTE *)(a2 + 8) )
    {
      case 0:
      case 1:
        return 16;
      case 2:
        return 32;
      case 3:
        return 64;
      case 4:
        return 80;
      case 5:
      case 6:
        return 128;
      case 8:
        v5 = 0;
        return *(unsigned int *)(sub_AE2980(a1, v5) + 4);
      case 0xA:
        return 0x2000;
      case 0xC:
        return *(_DWORD *)(a2 + 8) >> 8;
      case 0xE:
        v5 = *(_DWORD *)(a2 + 8) >> 8;
        return *(unsigned int *)(sub_AE2980(a1, v5) + 4);
      case 0xF:
        return 8LL * *(_QWORD *)sub_AE4AC0(a1, a2);
      case 0x10:
        v6 = *(_QWORD *)(a2 + 24);
        v7 = sub_AE5020(a1, v6);
        return 8
             * *(_QWORD *)(a2 + 32)
             * (((1LL << v7) + ((unsigned __int64)(sub_9208B0(a1, v6) + 7) >> 3) - 1) >> v7 << v7);
      case 0x11:
      case 0x12:
        v3 = *(unsigned int *)(a2 + 32);
        return sub_9208B0(a1, *(_QWORD *)(a2 + 24)) * v3;
      case 0x14:
        a2 = sub_BCE9B0(a2);
        break;
      default:
        BUG();
    }
  }
}
