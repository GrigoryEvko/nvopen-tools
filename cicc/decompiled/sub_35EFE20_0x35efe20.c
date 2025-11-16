// Function: sub_35EFE20
// Address: 0x35efe20
//
unsigned __int64 __fastcall sub_35EFE20(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v4; // rdx
  unsigned __int64 result; // rax
  _DWORD *v6; // rdx
  _DWORD *v7; // rdx
  _DWORD *v8; // rdx
  _DWORD *v9; // rdx
  _DWORD *v10; // rdx

  switch ( *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8) )
  {
    case 1LL:
      v6 = *(_DWORD **)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v6;
      if ( result <= 3 )
      {
        result = sub_CB6200(a4, ".f4e", 4u);
      }
      else
      {
        *v6 = 1697932846;
        *(_QWORD *)(a4 + 32) += 4LL;
      }
      break;
    case 2LL:
      v7 = *(_DWORD **)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v7;
      if ( result <= 3 )
      {
        result = sub_CB6200(a4, ".b4e", 4u);
      }
      else
      {
        *v7 = 1697931822;
        *(_QWORD *)(a4 + 32) += 4LL;
      }
      break;
    case 3LL:
      v8 = *(_DWORD **)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v8;
      if ( result <= 3 )
      {
        result = sub_CB6200(a4, ".rc8", 4u);
      }
      else
      {
        *v8 = 946041390;
        *(_QWORD *)(a4 + 32) += 4LL;
      }
      break;
    case 4LL:
      v9 = *(_DWORD **)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v9;
      if ( result <= 3 )
      {
        result = sub_CB6200(a4, ".ecl", 4u);
      }
      else
      {
        *v9 = 1818453294;
        *(_QWORD *)(a4 + 32) += 4LL;
      }
      break;
    case 5LL:
      v10 = *(_DWORD **)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v10;
      if ( result <= 3 )
      {
        result = sub_CB6200(a4, ".ecr", 4u);
      }
      else
      {
        *v10 = 1919116590;
        *(_QWORD *)(a4 + 32) += 4LL;
      }
      break;
    case 6LL:
      v4 = *(_QWORD *)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - v4;
      if ( result <= 4 )
      {
        result = sub_CB6200(a4, ".rc16", 5u);
      }
      else
      {
        *(_DWORD *)v4 = 828600878;
        *(_BYTE *)(v4 + 4) = 54;
        *(_QWORD *)(a4 + 32) += 5LL;
      }
      break;
    default:
      return result;
  }
  return result;
}
