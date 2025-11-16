// Function: sub_3737A60
// Address: 0x3737a60
//
__int64 __fastcall sub_3737A60(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax

  sub_37378D0(a1, a4, 2, *(_DWORD *)a2);
  if ( *(_BYTE *)(a2 + 5) )
    sub_3249A20(a1, (unsigned __int64 **)(a4 + 8), 15875, 65547, *(unsigned __int8 *)(a2 + 4));
  result = (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1[23] + 200) + 544LL) - 42);
  if ( (unsigned int)result <= 1 )
  {
    result = a1[26];
    if ( *(_DWORD *)(result + 6224) == 1 )
    {
      if ( *(_BYTE *)(a3 + 89) )
        return sub_3249A20(a1, (unsigned __int64 **)(a4 + 8), 51, 65547, 2);
    }
  }
  return result;
}
