// Function: sub_9C6480
// Address: 0x9c6480
//
__int64 __fastcall sub_9C6480(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v5; // rsi
  __int64 v6; // r14
  char v7; // r12

  while ( 1 )
  {
    switch ( *(_BYTE *)(a2 + 8) )
    {
      case 0:
      case 1:
        return 2;
      case 2:
        return 4;
      case 3:
        return 8;
      case 4:
        return 10;
      case 5:
      case 6:
        return 16;
      case 8:
        v5 = 0;
        return ((unsigned __int64)*(unsigned int *)(sub_AE2980(a1, v5) + 4) + 7) >> 3;
      case 0xA:
        return 1024;
      case 0xC:
        return ((unsigned __int64)(*(_DWORD *)(a2 + 8) >> 8) + 7) >> 3;
      case 0xE:
        v5 = *(_DWORD *)(a2 + 8) >> 8;
        return ((unsigned __int64)*(unsigned int *)(sub_AE2980(a1, v5) + 4) + 7) >> 3;
      case 0xF:
        return *(_QWORD *)sub_AE4AC0(a1, a2) & 0x1FFFFFFFFFFFFFFFLL;
      case 0x10:
        v6 = *(_QWORD *)(a2 + 24);
        v7 = sub_AE5020(a1, v6);
        return (8
              * *(_QWORD *)(a2 + 32)
              * (((1LL << v7) + ((unsigned __int64)(sub_9208B0(a1, v6) + 7) >> 3) - 1) >> v7 << v7)
              + 7) >> 3;
      case 0x11:
      case 0x12:
        v3 = *(unsigned int *)(a2 + 32);
        return (unsigned __int64)(sub_9208B0(a1, *(_QWORD *)(a2 + 24)) * v3 + 7) >> 3;
      case 0x14:
        a2 = sub_BCE9B0(a2);
        break;
      default:
        BUG();
    }
  }
}
