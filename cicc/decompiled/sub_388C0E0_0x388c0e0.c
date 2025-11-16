// Function: sub_388C0E0
// Address: 0x388c0e0
//
__int64 __fastcall sub_388C0E0(__int64 a1, _BYTE *a2)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 64);
  if ( (_DWORD)result == 24 )
  {
    *a2 = 1;
    result = sub_3887100(a1 + 8);
    *(_DWORD *)(a1 + 64) = result;
  }
  else
  {
    *a2 = 0;
    if ( (_DWORD)result == 25 )
    {
      result = sub_3887100(a1 + 8);
      *(_DWORD *)(a1 + 64) = result;
    }
  }
  return result;
}
