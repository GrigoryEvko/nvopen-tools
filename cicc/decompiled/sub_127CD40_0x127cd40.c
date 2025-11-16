// Function: sub_127CD40
// Address: 0x127cd40
//
void __fastcall sub_127CD40(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // rax

  while ( 2 )
  {
    switch ( *(_BYTE *)(a2 + 40) )
    {
      case 0:
      case 0xF:
      case 0x11:
      case 0x12:
      case 0x18:
        return;
      case 1:
        sub_127CFB0();
        return;
      case 2:
        v2 = *(_QWORD *)(a2 + 72);
        a2 = *(_QWORD *)(v2 + 8);
        if ( (*(_BYTE *)(v2 + 24) & 2) != 0 )
          a2 = *(_QWORD *)v2;
        if ( a2 )
          continue;
        nullsub_2014();
        return;
      case 5:
        a2 = *(_QWORD *)(a2 + 72);
        continue;
      case 6:
        sub_127CA00(a1, a2);
        return;
      case 7:
        sub_127CA80(a1, a2);
        return;
      case 8:
        sub_127CAB0(a1, a2);
        return;
      case 0xB:
        sub_127CF10();
        return;
      case 0xC:
        a2 = *(_QWORD *)(a2 + 72);
        continue;
      case 0xD:
        a2 = *(_QWORD *)(a2 + 72);
        continue;
      case 0x10:
        sub_127CB50(a1, a2);
        return;
      case 0x14:
        sub_127CAE0(a1, a2);
        return;
      default:
        sub_127B550("unsupported statement type", (_DWORD *)a2, 1);
    }
  }
}
