// Function: sub_1C30980
// Address: 0x1c30980
//
bool __fastcall sub_1C30980(__int64 a1)
{
  bool result; // al
  __int64 v3; // rax
  unsigned int v4; // edi
  __int64 v5; // rdx
  _QWORD *v6; // rax

  if ( *(_BYTE *)(a1 + 16) != 78 )
    return 0;
  v3 = *(_QWORD *)(a1 - 24);
  if ( *(_BYTE *)(v3 + 16) || (*(_BYTE *)(v3 + 33) & 0x20) == 0 )
    return 0;
  v4 = *(_DWORD *)(v3 + 36);
  if ( v4 <= 0x141C )
  {
    if ( v4 > 0x13DF )
    {
      switch ( v4 )
      {
        case 0x13E0u:
        case 0x13E2u:
        case 0x13E6u:
        case 0x13E8u:
        case 0x13F0u:
        case 0x13F2u:
        case 0x13F6u:
        case 0x13F8u:
        case 0x13FEu:
        case 0x1400u:
        case 0x1406u:
        case 0x1408u:
        case 0x140Cu:
        case 0x140Eu:
        case 0x1410u:
        case 0x1412u:
        case 0x1414u:
        case 0x1416u:
        case 0x141Au:
        case 0x141Cu:
          return 0;
        case 0x1418u:
          result = 1;
          break;
        default:
          goto LABEL_10;
      }
      return result;
    }
    if ( v4 > 0xF04 )
    {
      switch ( v4 )
      {
        case 0x114Cu:
        case 0x114Eu:
        case 0x1152u:
        case 0x1154u:
        case 0x115Cu:
        case 0x115Eu:
        case 0x1162u:
        case 0x1164u:
        case 0x1168u:
        case 0x116Au:
        case 0x116Cu:
        case 0x116Eu:
        case 0x1170u:
        case 0x1172u:
        case 0x1174u:
          return 0;
        default:
          goto LABEL_10;
      }
    }
    result = 1;
    if ( v4 > 0xF00 )
      return result;
  }
LABEL_10:
  if ( !sub_1C30440(v4) )
    return 0;
  v5 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v6 = *(_QWORD **)(v5 + 24);
  if ( *(_DWORD *)(v5 + 32) > 0x40u )
    v6 = (_QWORD *)*v6;
  return ((unsigned __int8)v6 & 0x30) != 32;
}
