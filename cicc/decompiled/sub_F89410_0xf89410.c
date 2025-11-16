// Function: sub_F89410
// Address: 0xf89410
//
__int64 __fastcall sub_F89410(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax

  switch ( *(_WORD *)(a2 + 24) )
  {
    case 0:
      result = *(_QWORD *)(a2 + 32);
      break;
    case 1:
      result = sub_F7DFA0(a1, a2);
      break;
    case 2:
      result = sub_F8D160(a1);
      break;
    case 3:
      result = sub_F8D260(a1);
      break;
    case 4:
      result = sub_F8D3A0(a1);
      break;
    case 5:
      result = sub_F8A680(a1);
      break;
    case 6:
      result = sub_F8DD40(a1);
      break;
    case 7:
      result = sub_F8AA70(a1);
      break;
    case 8:
      if ( *(_BYTE *)(a1 + 512) && *(_QWORD *)(a2 + 40) <= 2u )
        result = sub_F8C4D0();
      else
        result = sub_F88DC0(a1, a2, a3, a4, a5);
      break;
    case 9:
      result = sub_F8DA50(a1);
      break;
    case 0xA:
      result = sub_F8DA10(a1);
      break;
    case 0xB:
      result = sub_F8DAD0(a1);
      break;
    case 0xC:
      result = sub_F8DA90(a1);
      break;
    case 0xD:
      result = sub_F8DB10(a1);
      break;
    case 0xE:
      result = sub_F8D110(a1);
      break;
    case 0xF:
      result = *(_QWORD *)(a2 - 8);
      break;
    default:
      BUG();
  }
  return result;
}
