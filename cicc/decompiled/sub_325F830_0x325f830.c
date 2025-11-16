// Function: sub_325F830
// Address: 0x325f830
//
__int64 __fastcall sub_325F830(unsigned __int8 **a1, __int64 a2)
{
  __int64 result; // rax

  result = 0;
  if ( *(_DWORD *)(a2 + 24) == 98 )
  {
    result = **a1;
    if ( (_BYTE)result || (*(_BYTE *)(a2 + 29) & 2) != 0 )
    {
      result = 1;
      if ( (*(_BYTE *)(*(_QWORD *)a1[1] + 8LL) & 1) == 0 )
        return (*(_DWORD *)(a2 + 28) >> 11) & 1;
    }
  }
  return result;
}
