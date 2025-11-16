// Function: sub_270B150
// Address: 0x270b150
//
__int64 __fastcall sub_270B150(__int64 a1, int a2)
{
  __int64 result; // rax

  switch ( a2 )
  {
    case 0:
      result = *(_QWORD *)(a1 + 8);
      if ( !result )
      {
        result = sub_B6E160(*(__int64 **)a1, 0x102u, 0, 0);
        *(_QWORD *)(a1 + 8) = result;
      }
      break;
    case 1:
      result = *(_QWORD *)(a1 + 16);
      if ( !result )
      {
        result = sub_B6E160(*(__int64 **)a1, 0x10Bu, 0, 0);
        *(_QWORD *)(a1 + 16) = result;
      }
      break;
    case 2:
      result = *(_QWORD *)(a1 + 24);
      if ( !result )
      {
        result = sub_B6E160(*(__int64 **)a1, 0x10Cu, 0, 0);
        *(_QWORD *)(a1 + 24) = result;
      }
      break;
    case 3:
      result = *(_QWORD *)(a1 + 32);
      if ( !result )
      {
        result = sub_B6E160(*(__int64 **)a1, 0x111u, 0, 0);
        *(_QWORD *)(a1 + 32) = result;
      }
      break;
    case 4:
      result = *(_QWORD *)(a1 + 40);
      if ( !result )
      {
        result = sub_B6E160(*(__int64 **)a1, 0xFFu, 0, 0);
        *(_QWORD *)(a1 + 40) = result;
      }
      break;
    case 5:
      result = *(_QWORD *)(a1 + 48);
      if ( !result )
      {
        result = sub_B6E160(*(__int64 **)a1, 0x113u, 0, 0);
        *(_QWORD *)(a1 + 48) = result;
      }
      break;
    case 6:
      result = *(_QWORD *)(a1 + 56);
      if ( !result )
      {
        result = sub_B6E160(*(__int64 **)a1, 0x110u, 0, 0);
        *(_QWORD *)(a1 + 56) = result;
      }
      break;
    case 7:
      result = *(_QWORD *)(a1 + 64);
      if ( !result )
      {
        result = sub_B6E160(*(__int64 **)a1, 0x119u, 0, 0);
        *(_QWORD *)(a1 + 64) = result;
      }
      break;
    case 8:
      result = *(_QWORD *)(a1 + 72);
      if ( !result )
      {
        result = sub_B6E160(*(__int64 **)a1, 0x10Eu, 0, 0);
        *(_QWORD *)(a1 + 72) = result;
      }
      break;
    case 9:
      result = *(_QWORD *)(a1 + 80);
      if ( !result )
      {
        result = sub_B6E160(*(__int64 **)a1, 0x10Fu, 0, 0);
        *(_QWORD *)(a1 + 80) = result;
      }
      break;
    default:
      BUG();
  }
  return result;
}
