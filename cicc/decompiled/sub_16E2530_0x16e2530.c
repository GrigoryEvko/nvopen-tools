// Function: sub_16E2530
// Address: 0x16e2530
//
__int64 __fastcall sub_16E2530(__int64 a1, unsigned int *a2, _DWORD *a3, _DWORD *a4)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 44);
  if ( (_DWORD)result == 11 || (_DWORD)result == 3 )
  {
    *a2 = 5;
    *a3 = 0;
    *a4 = 0;
  }
  else
  {
    sub_16E2390(a1, a2, a3, a4);
    result = *a2;
    if ( !(_DWORD)result )
    {
      result = 2 * (unsigned int)(*(_DWORD *)(a1 + 32) == 3) + 5;
      *a2 = result;
    }
  }
  return result;
}
