// Function: sub_120C370
// Address: 0x120c370
//
__int64 __fastcall sub_120C370(__int64 a1, _BYTE *a2)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 240);
  if ( (_DWORD)result == 26 )
  {
    *a2 = 1;
    result = sub_1205200(a1 + 176);
    *(_DWORD *)(a1 + 240) = result;
  }
  else
  {
    *a2 = 0;
    if ( (_DWORD)result == 27 )
    {
      result = sub_1205200(a1 + 176);
      *(_DWORD *)(a1 + 240) = result;
    }
  }
  return result;
}
