// Function: sub_3206290
// Address: 0x3206290
//
__int64 __fastcall sub_3206290(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 result; // rax
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx

  switch ( (unsigned __int16)sub_AF18C0((__int64)a2) )
  {
    case 1u:
      result = sub_32067D0(a1, a2);
      break;
    case 2u:
    case 0x13u:
      result = sub_3205AE0(a1, a2);
      break;
    case 4u:
      result = sub_32080C0(a1, a2);
      break;
    case 0xFu:
      v7 = sub_A547D0((__int64)a2, 2);
      if ( v11 != 15
        || *(_QWORD *)v7 != 0x705F6C6274765F5FLL
        || *(_DWORD *)(v7 + 8) != 1952412276
        || *(_WORD *)(v7 + 12) != 28793
        || *(_BYTE *)(v7 + 14) != 101 )
      {
        goto LABEL_6;
      }
      result = sub_31F8400(a1, (__int64)a2, 0x705F6C6274765F5FLL, v8, v9, v10);
      break;
    case 0x10u:
    case 0x42u:
LABEL_6:
      result = sub_3206C30(a1, a2, 0);
      break;
    case 0x12u:
      result = sub_31F7FC0(a1, (__int64)a2);
      break;
    case 0x15u:
      if ( a3 )
        result = sub_3206EF0(a1, a2, a3, 0, 0, 0);
      else
        result = sub_3207D60(a1, a2);
      break;
    case 0x16u:
      result = sub_3206680(a1, a2);
      break;
    case 0x17u:
      result = sub_32060E0(a1, a2);
      break;
    case 0x1Fu:
      result = sub_3207960(a1, a2, 0);
      break;
    case 0x24u:
      result = sub_31F80D0(a1, (__int64)a2);
      break;
    case 0x26u:
    case 0x35u:
    case 0x37u:
      result = sub_3207BD0(a1, a2);
      break;
    case 0x3Bu:
      v4 = sub_A547D0((__int64)a2, 2);
      if ( v5 != 17
        || *(_QWORD *)v4 ^ 0x657079746C636564LL | *(_QWORD *)(v4 + 8) ^ 0x7274706C6C756E28LL
        || *(_BYTE *)(v4 + 16) != 41 )
      {
        goto LABEL_3;
      }
      result = 259;
      break;
    default:
LABEL_3:
      result = 0;
      break;
  }
  return result;
}
