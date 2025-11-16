// Function: sub_20D63A0
// Address: 0x20d63a0
//
__int64 __fastcall sub_20D63A0(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax

  result = 0xFFFFFFFFLL;
  if ( *(_DWORD *)a1 >= *(_DWORD *)a2 )
  {
    result = 1;
    if ( *(_DWORD *)a1 <= *(_DWORD *)a2 )
      return 2 * (unsigned int)(*(_DWORD *)(a1[1] + 48LL) >= *(_DWORD *)(a2[1] + 48LL)) - 1;
  }
  return result;
}
