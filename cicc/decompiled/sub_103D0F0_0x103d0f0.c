// Function: sub_103D0F0
// Address: 0x103d0f0
//
unsigned __int64 *__fastcall sub_103D0F0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 *result; // rax
  bool v5; // zf

  result = sub_103CDC0(a1, a2, 0);
  if ( *(_BYTE *)a2 == 27 )
  {
    v5 = *(_QWORD *)(a2 - 32) == 0;
    *(_DWORD *)(a2 + 84) = -1;
    if ( !v5 )
    {
      result = *(unsigned __int64 **)(a2 - 24);
      **(_QWORD **)(a2 - 16) = result;
      if ( result )
        result[2] = *(_QWORD *)(a2 - 16);
    }
    *(_QWORD *)(a2 - 32) = 0;
  }
  *(_QWORD *)(a2 + 64) = a3;
  return result;
}
