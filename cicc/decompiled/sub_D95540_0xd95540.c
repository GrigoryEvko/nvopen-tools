// Function: sub_D95540
// Address: 0xd95540
//
__int64 __fastcall sub_D95540(__int64 a1)
{
  while ( 2 )
  {
    switch ( *(_WORD *)(a1 + 24) )
    {
      case 0:
        return *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL);
      case 1:
        return *(_QWORD *)(a1 + 32);
      case 2:
      case 3:
      case 4:
      case 0xE:
        return *(_QWORD *)(a1 + 40);
      case 5:
        return *(_QWORD *)(a1 + 48);
      case 6:
      case 8:
      case 9:
      case 0xA:
      case 0xB:
      case 0xC:
      case 0xD:
        a1 = **(_QWORD **)(a1 + 32);
        continue;
      case 7:
        a1 = *(_QWORD *)(a1 + 40);
        continue;
      case 0xF:
        return *(_QWORD *)(*(_QWORD *)(a1 - 8) + 8LL);
      default:
        BUG();
    }
  }
}
