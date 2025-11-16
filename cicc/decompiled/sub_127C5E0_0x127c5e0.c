// Function: sub_127C5E0
// Address: 0x127c5e0
//
void __fastcall sub_127C5E0(__int64 a1, _DWORD *a2)
{
  char *v2; // rax

  if ( a1 )
  {
    if ( !*(_BYTE *)(a1 + 174) )
    {
      switch ( *(_WORD *)(a1 + 176) )
      {
        case 0x6241:
        case 0x6242:
        case 0x6248:
        case 0x6249:
        case 0x624F:
        case 0x6250:
        case 0x6257:
        case 0x6258:
        case 0x625F:
        case 0x6260:
        case 0x6263:
        case 0x6264:
        case 0x6267:
        case 0x6268:
        case 0x626B:
        case 0x626C:
        case 0x6273:
        case 0x6274:
        case 0x627B:
        case 0x627C:
        case 0x6280:
        case 0x6281:
        case 0x6286:
          sub_6851C0(0xEB8u, a2);
          break;
        default:
          break;
      }
    }
    if ( unk_4D046E8
      && HIDWORD(qword_4D045BC)
      && (*(_BYTE *)(a1 + 193) & 0x10) == 0
      && *(_BYTE *)(a1 + 172) == 2
      && *(_QWORD *)(a1 + 8)
      && (*(_BYTE *)(a1 + 89) & 4) == 0 )
    {
      v2 = sub_8258E0(a1, 0);
      sub_684B10(0xE9Eu, a2, (__int64)v2);
    }
  }
}
