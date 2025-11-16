// Function: sub_11F17D0
// Address: 0x11f17d0
//
__int64 __fastcall sub_11F17D0(__int64 a1, __int64 a2, unsigned int **a3)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 *v7; // r14
  __int64 result; // rax
  unsigned int v9[13]; // [rsp+Ch] [rbp-34h] BYREF

  v5 = sub_B43CA0(a2);
  v6 = *(_QWORD *)(a2 - 32);
  v7 = (__int64 *)v5;
  if ( v6 )
  {
    if ( *(_BYTE *)v6 )
    {
      v6 = 0;
    }
    else if ( *(_QWORD *)(v6 + 24) != *(_QWORD *)(a2 + 80) )
    {
      v6 = 0;
    }
  }
  if ( !sub_981210(**(_QWORD **)(a1 + 24), v6, v9) || !sub_11C99B0(v7, *(__int64 **)(a1 + 24), v9[0]) || v9[0] > 0x209 )
    return 0;
  if ( v9[0] <= 0x162 )
  {
    if ( v9[0] == 186 )
      return sub_11E3330(a1, a2, a3);
    if ( v9[0] > 0xBA )
    {
      if ( v9[0] == 187 )
        return sub_11E9AC0(a1, a2, (__int64)a3);
      return 0;
    }
    if ( v9[0] > 0x31 )
    {
      if ( v9[0] - 54 > 0xB )
        return 0;
    }
    else if ( v9[0] <= 0x29 )
    {
      return 0;
    }
    return sub_11E3B10(a1, a2, (__int64)a3, v9);
  }
  switch ( v9[0] )
  {
    case 0x163u:
      result = sub_11E3450(a1, a2, (__int64)a3);
      break;
    case 0x164u:
      result = sub_11E14F0((_QWORD *)a1, a2, (__int64)a3);
      break;
    case 0x165u:
      result = sub_11E3240(a1, a2, a3);
      break;
    case 0x166u:
      result = sub_11E3340(a1, a2, (__int64)a3);
      break;
    case 0x167u:
      result = sub_11E3820(a1, a2, (__int64)a3);
      break;
    case 0x168u:
      result = sub_11E3750(a1, a2, (__int64)a3);
      break;
    case 0x169u:
      result = sub_11E0B50(a1, a2, a3);
      break;
    case 0x16Au:
      result = sub_11E3930(a1, a2, (__int64 *)a3);
      break;
    case 0x190u:
      result = sub_11E3A90(a1, a2, (__int64)a3);
      break;
    case 0x1C8u:
      result = sub_11DF4F0(a1, a2, (__int64)a3);
      break;
    case 0x1C9u:
      result = sub_11DFDD0(a1, a2, 1, (__int64)a3);
      break;
    case 0x1CBu:
      result = sub_11DD610(a1, a2, (__int64)a3);
      break;
    case 0x1CCu:
      result = sub_11DD8F0(a1, a2, (__int64)a3);
      break;
    case 0x1CDu:
      result = sub_11DDEE0(a1, a2, (__int64)a3);
      break;
    case 0x1CFu:
      result = sub_11DF3B0(a1, a2, (__int64)a3);
      break;
    case 0x1D0u:
      result = sub_11E09F0(a1, a2, (__int64)a3);
      break;
    case 0x1D3u:
      result = sub_11DF750(a1, a2, (__int64)a3);
      break;
    case 0x1D4u:
      result = sub_11F1630(a1, a2, (__int64)a3);
      break;
    case 0x1D6u:
      result = sub_11DD710(a1, a2, (__int64)a3);
      break;
    case 0x1D7u:
      result = sub_11DE8A0(a1, a2, (__int64)a3);
      break;
    case 0x1D8u:
      result = sub_11DFDD0(a1, a2, 0, (__int64)a3);
      break;
    case 0x1D9u:
      result = sub_11DF2C0(a1, a2, (__int64)a3);
      break;
    case 0x1DAu:
      result = sub_11F1680(a1, a2, (__int64)a3);
      break;
    case 0x1DBu:
      result = sub_11E06A0(a1, a2, (__int64)a3);
      break;
    case 0x1DCu:
      result = sub_11DDD00(a1, a2, (__int64)a3);
      break;
    case 0x1DDu:
      result = sub_11E0900(a1, a2);
      break;
    case 0x1DEu:
      result = sub_11EA710(a1, a2, (__int64)a3);
      break;
    case 0x1DFu:
    case 0x1E0u:
    case 0x1E3u:
    case 0x1E4u:
    case 0x1E5u:
    case 0x1E6u:
    case 0x1E7u:
      result = sub_11E0870(a1, a2);
      break;
    case 0x209u:
      result = sub_11F1760(a1, a2, (__int64)a3);
      break;
    default:
      return 0;
  }
  return result;
}
