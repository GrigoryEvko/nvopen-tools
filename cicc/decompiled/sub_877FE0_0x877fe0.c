// Function: sub_877FE0
// Address: 0x877fe0
//
__int64 __fastcall sub_877FE0(__int64 a1)
{
  __int64 v1; // r12

  switch ( *(_BYTE *)(a1 + 80) )
  {
    case 3:
      v1 = *(_QWORD *)(a1 + 88);
      break;
    case 7:
    case 8:
    case 9:
      v1 = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 120LL);
      break;
    case 0xA:
    case 0xB:
      v1 = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 152LL);
      break;
    case 0x14:
      v1 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 88) + 176LL) + 152LL);
      break;
    default:
      return 0;
  }
LABEL_6:
  if ( !v1 )
    return 0;
  while ( !(unsigned int)sub_8D2310(v1) )
  {
    while ( *(_BYTE *)(v1 + 140) == 12 )
      v1 = *(_QWORD *)(v1 + 160);
    if ( (unsigned int)sub_8D3320(v1) )
    {
      v1 = sub_8D46C0(v1);
      goto LABEL_6;
    }
    if ( (unsigned int)sub_8D3D10(v1) )
    {
      v1 = sub_8D4870(v1);
      goto LABEL_6;
    }
    if ( (unsigned int)sub_8D3410(v1) )
    {
      v1 = sub_8D4050(v1);
      goto LABEL_6;
    }
    if ( !(unsigned int)sub_8D2310(v1) )
      return 0;
  }
  return v1;
}
