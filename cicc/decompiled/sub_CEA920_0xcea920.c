// Function: sub_CEA920
// Address: 0xcea920
//
bool __fastcall sub_CEA920(__int64 a1)
{
  bool result; // al
  __int64 v3; // rax
  unsigned int v4; // edi
  __int64 v5; // rdx
  _QWORD *v6; // rax

  if ( *(_BYTE *)a1 != 85 )
    return 0;
  v3 = *(_QWORD *)(a1 - 32);
  if ( !v3 || *(_BYTE *)v3 || *(_QWORD *)(v3 + 24) != *(_QWORD *)(a1 + 80) || (*(_BYTE *)(v3 + 33) & 0x20) == 0 )
    return 0;
  v4 = *(_DWORD *)(v3 + 36);
  if ( v4 > 0x28FE )
  {
LABEL_14:
    if ( sub_CEA3C0(v4) )
    {
      v5 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
      v6 = *(_QWORD **)(v5 + 24);
      if ( *(_DWORD *)(v5 + 32) > 0x40u )
        v6 = (_QWORD *)*v6;
      return ((unsigned __int8)v6 & 0x30) != 32;
    }
    return 0;
  }
  if ( v4 > 0x28C1 )
  {
    switch ( v4 )
    {
      case 0x28C2u:
      case 0x28C4u:
      case 0x28C8u:
      case 0x28CAu:
      case 0x28D2u:
      case 0x28D4u:
      case 0x28D8u:
      case 0x28DAu:
      case 0x28E0u:
      case 0x28E2u:
      case 0x28E8u:
      case 0x28EAu:
      case 0x28EEu:
      case 0x28F0u:
      case 0x28F2u:
      case 0x28F4u:
      case 0x28F6u:
      case 0x28F8u:
      case 0x28FCu:
      case 0x28FEu:
        return 0;
      case 0x28FAu:
        result = 1;
        break;
      default:
        goto LABEL_14;
    }
  }
  else
  {
    if ( v4 > 0x2126 )
    {
      switch ( v4 )
      {
        case 0x250Cu:
        case 0x250Eu:
        case 0x2512u:
        case 0x2514u:
        case 0x251Cu:
        case 0x251Eu:
        case 0x2522u:
        case 0x2524u:
        case 0x2528u:
        case 0x252Au:
        case 0x252Cu:
        case 0x252Eu:
        case 0x2530u:
        case 0x2532u:
        case 0x2534u:
          return 0;
        default:
          goto LABEL_14;
      }
    }
    result = 1;
    if ( v4 <= 0x2122 )
      goto LABEL_14;
  }
  return result;
}
