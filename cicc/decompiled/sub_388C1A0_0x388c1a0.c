// Function: sub_388C1A0
// Address: 0x388c1a0
//
__int64 __fastcall sub_388C1A0(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 64);
  if ( (_DWORD)result == 33 )
  {
    *a2 = 1;
  }
  else
  {
    if ( (_DWORD)result != 34 )
    {
      *a2 = 0;
      return result;
    }
    *a2 = 2;
  }
  result = sub_3887100(a1 + 8);
  *(_DWORD *)(a1 + 64) = result;
  return result;
}
