// Function: sub_120C4A0
// Address: 0x120c4a0
//
__int64 __fastcall sub_120C4A0(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 240);
  if ( (_DWORD)result == 35 )
  {
    *a2 = 1;
  }
  else
  {
    if ( (_DWORD)result != 36 )
    {
      *a2 = 0;
      return result;
    }
    *a2 = 2;
  }
  result = sub_1205200(a1 + 176);
  *(_DWORD *)(a1 + 240) = result;
  return result;
}
