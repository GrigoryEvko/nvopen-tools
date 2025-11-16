// Function: sub_893570
// Address: 0x893570
//
__int64 __fastcall sub_893570(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rax

  result = dword_4D047B0;
  if ( !dword_4D047B0 )
  {
    switch ( *(_BYTE *)(a1 + 80) )
    {
      case 4:
      case 5:
        v2 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
        goto LABEL_5;
      case 6:
        v2 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
        goto LABEL_5;
      case 9:
      case 0xA:
        return (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL) + 160LL) & 8) != 0;
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
        v2 = *(_QWORD *)(a1 + 88);
LABEL_5:
        result = (*(_BYTE *)(v2 + 160) & 8) != 0;
        break;
      default:
        BUG();
    }
  }
  return result;
}
