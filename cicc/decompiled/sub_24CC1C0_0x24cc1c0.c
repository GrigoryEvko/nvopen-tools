// Function: sub_24CC1C0
// Address: 0x24cc1c0
//
__int64 __fastcall sub_24CC1C0(__int64 a1, int a2)
{
  __int64 v2; // r12
  __int64 v3; // rax

  switch ( a2 )
  {
    case 0:
      BUG();
    case 4:
      v2 = 2;
      break;
    case 5:
      v2 = 3;
      break;
    case 6:
      v2 = 4;
      break;
    case 7:
      v2 = 5;
      break;
    default:
      v2 = 0;
      break;
  }
  v3 = sub_BCB2D0(*(_QWORD **)(a1 + 72));
  return sub_ACD640(v3, v2, 0);
}
