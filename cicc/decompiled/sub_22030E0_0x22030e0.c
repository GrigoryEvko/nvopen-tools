// Function: sub_22030E0
// Address: 0x22030e0
//
__int64 __fastcall sub_22030E0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rdx

  switch ( **(_WORD **)(a1 + 16) )
  {
    case 0xB8D:
    case 0xB8E:
    case 0xB99:
    case 0xB9A:
    case 0xBA5:
    case 0xBA6:
    case 0xBB1:
    case 0xBB2:
    case 0xBBD:
    case 0xBBE:
    case 0xBC9:
    case 0xBCA:
      result = 7;
      v2 = 3;
      goto LABEL_3;
    case 0xB93:
    case 0xB94:
    case 0xB9F:
    case 0xBA0:
    case 0xBAB:
    case 0xBAC:
    case 0xBB7:
    case 0xBB8:
    case 0xBC3:
    case 0xBC4:
    case 0xBCF:
    case 0xBD0:
      result = 9;
      v2 = 5;
      goto LABEL_3;
    case 0xBE1:
    case 0xBE2:
    case 0xBE7:
    case 0xBE8:
    case 0xBED:
    case 0xBEE:
    case 0xBF3:
    case 0xBF4:
    case 0xBF9:
    case 0xBFA:
    case 0xBFF:
    case 0xC00:
      result = 6;
      v2 = 2;
LABEL_3:
      if ( (unsigned int)result >= *(_DWORD *)(a1 + 40) )
        goto LABEL_5;
      v3 = *(_QWORD *)(a1 + 32) + 40 * v2;
      if ( *(_BYTE *)v3 != 1 )
        goto LABEL_5;
      if ( *(_QWORD *)(v3 + 24) != 4 )
        result = 0;
      break;
    default:
LABEL_5:
      result = 0;
      break;
  }
  return result;
}
