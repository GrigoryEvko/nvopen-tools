// Function: sub_8D48B0
// Address: 0x8d48b0
//
__int64 __fastcall sub_8D48B0(__int64 a1, int *a2)
{
  __int64 result; // rax
  int v3; // edx

  switch ( *(_BYTE *)(a1 + 140) )
  {
    case 6:
      result = sub_8D46C0(a1);
      v3 = 1;
      break;
    case 7:
    case 0xC:
      result = *(_QWORD *)(a1 + 160);
      v3 = 1;
      break;
    case 8:
      result = sub_8D4050(a1);
      v3 = 1;
      break;
    case 0xD:
      result = sub_8D4870(a1);
      v3 = 1;
      break;
    default:
      v3 = 0;
      result = 0;
      break;
  }
  if ( a2 )
    *a2 = v3;
  return result;
}
