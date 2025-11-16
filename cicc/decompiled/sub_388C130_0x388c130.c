// Function: sub_388C130
// Address: 0x388c130
//
__int64 __fastcall sub_388C130(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 64);
  if ( (_DWORD)result == 38 )
  {
    *a2 = 1;
    result = sub_3887100(a1 + 8);
    *(_DWORD *)(a1 + 64) = result;
    return result;
  }
  if ( (_DWORD)result == 39 )
  {
    *a2 = 2;
    goto LABEL_6;
  }
  *a2 = 0;
  if ( (_DWORD)result == 37 )
  {
LABEL_6:
    result = sub_3887100(a1 + 8);
    *(_DWORD *)(a1 + 64) = result;
  }
  return result;
}
