// Function: sub_1E0A730
// Address: 0x1e0a730
//
__int64 __fastcall sub_1E0A730(_DWORD *a1, __int64 a2)
{
  __int64 result; // rax

  switch ( *a1 )
  {
    case 0:
      result = sub_15A9520(a2, 0);
      break;
    case 1:
      result = 8;
      break;
    case 2:
    case 3:
    case 5:
      result = 4;
      break;
    case 4:
      result = 0;
      break;
  }
  return result;
}
