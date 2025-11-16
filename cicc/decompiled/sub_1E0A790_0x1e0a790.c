// Function: sub_1E0A790
// Address: 0x1e0a790
//
__int64 __fastcall sub_1E0A790(_DWORD *a1, __int64 a2)
{
  __int64 result; // rax

  switch ( *a1 )
  {
    case 0:
      result = sub_15A9480(a2, 0);
      break;
    case 1:
      result = sub_15AAE10(a2, 0x40u);
      break;
    case 2:
    case 3:
    case 5:
      result = sub_15AAE10(a2, 0x20u);
      break;
    case 4:
      result = 1;
      break;
  }
  return result;
}
