// Function: sub_1602570
// Address: 0x1602570
//
__int64 __fastcall sub_1602570(__int64 a1)
{
  __int64 result; // rax

  if ( (unsigned int)(*(_DWORD *)(a1 + 8) - 8) > 8 )
    return 1;
  result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  if ( (_BYTE)result )
  {
    result = 1;
    if ( *(_BYTE *)(a1 + 456) )
      return *(unsigned __int8 *)(a1 + 80);
  }
  return result;
}
