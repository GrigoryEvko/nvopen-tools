// Function: sub_16E2590
// Address: 0x16e2590
//
__int64 __fastcall sub_16E2590(__int64 a1, _DWORD *a2, _DWORD *a3, _DWORD *a4)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 44);
  if ( (_DWORD)result == 30 )
  {
    sub_16E2390(a1, a2, a3, a4);
    result = (unsigned int)*a2;
    if ( !(_DWORD)result )
      *a2 = 2;
  }
  else
  {
    *a2 = 2;
    *a3 = 0;
    *a4 = 0;
  }
  return result;
}
