// Function: sub_1456040
// Address: 0x1456040
//
__int64 __fastcall sub_1456040(__int64 a1)
{
  __int64 result; // rax

  while ( 2 )
  {
    switch ( *(_WORD *)(a1 + 24) )
    {
      case 0:
        result = **(_QWORD **)(a1 + 32);
        break;
      case 1:
      case 2:
      case 3:
        result = *(_QWORD *)(a1 + 40);
        break;
      case 4:
        a1 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL * ((unsigned int)*(_QWORD *)(a1 + 40) - 1));
        continue;
      case 5:
      case 7:
      case 8:
      case 9:
        a1 = **(_QWORD **)(a1 + 32);
        continue;
      case 6:
        a1 = *(_QWORD *)(a1 + 40);
        continue;
      case 0xA:
        result = **(_QWORD **)(a1 - 8);
        break;
    }
    return result;
  }
}
