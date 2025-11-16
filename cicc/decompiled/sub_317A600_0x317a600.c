// Function: sub_317A600
// Address: 0x317a600
//
__int64 __fastcall sub_317A600(__int64 *a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 result; // rax

  switch ( *a2 )
  {
    case 0x1Eu:
    case 0x1Fu:
    case 0x20u:
    case 0x21u:
    case 0x23u:
    case 0x24u:
    case 0x25u:
    case 0x26u:
    case 0x27u:
    case 0x3Cu:
    case 0x3Eu:
    case 0x40u:
    case 0x41u:
    case 0x42u:
    case 0x50u:
    case 0x51u:
    case 0x57u:
    case 0x58u:
    case 0x59u:
    case 0x5Au:
    case 0x5Bu:
    case 0x5Cu:
    case 0x5Du:
    case 0x5Eu:
    case 0x5Fu:
      result = 0;
      break;
    case 0x22u:
    case 0x28u:
    case 0x55u:
      result = sub_3175680((__int64)a1, (__int64)a2);
      break;
    case 0x29u:
      result = sub_96E680((unsigned int)*a2 - 29, *(_QWORD *)(a1[30] + 8));
      break;
    case 0x2Au:
    case 0x2Bu:
    case 0x2Cu:
    case 0x2Du:
    case 0x2Eu:
    case 0x2Fu:
    case 0x30u:
    case 0x31u:
    case 0x32u:
    case 0x33u:
    case 0x34u:
    case 0x35u:
    case 0x36u:
    case 0x37u:
    case 0x38u:
    case 0x39u:
    case 0x3Au:
    case 0x3Bu:
      result = (__int64)sub_3175FF0((__int64)a1, a2);
      break;
    case 0x3Du:
      v6 = a1[30];
      if ( **(_BYTE **)(v6 + 8) == 20 )
        result = 0;
      else
        result = sub_9718F0(*(_QWORD *)(v6 + 8), *((_QWORD *)a2 + 1), (_BYTE *)a1[5]);
      break;
    case 0x3Fu:
      result = sub_31758E0((__int64)a1, (__int64)a2, a3, a4, a5, a6);
      break;
    case 0x43u:
    case 0x44u:
    case 0x45u:
    case 0x46u:
    case 0x47u:
    case 0x48u:
    case 0x49u:
    case 0x4Au:
    case 0x4Bu:
    case 0x4Cu:
    case 0x4Du:
    case 0x4Eu:
    case 0x4Fu:
      result = sub_96F480((unsigned int)*a2 - 29, *(_QWORD *)(a1[30] + 8), *((_QWORD *)a2 + 1), a1[5]);
      break;
    case 0x52u:
    case 0x53u:
      result = sub_3175B30(a1, (__int64)a2);
      break;
    case 0x54u:
      result = sub_317A130((__int64)a1, (__int64)a2);
      break;
    case 0x56u:
      result = sub_3175A30((__int64)a1, (__int64)a2, a3, a4, a5);
      break;
    case 0x60u:
      result = sub_3175640((__int64)a1);
      break;
    default:
      BUG();
  }
  return result;
}
