// Function: sub_B6E560
// Address: 0xb6e560
//
__int64 __fastcall sub_B6E560(__int64 a1)
{
  __int64 result; // rax

  if ( (unsigned int)(*(_DWORD *)(a1 + 8) - 13) > 8 )
    return 1;
  result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 32LL))(a1);
  if ( (_BYTE)result )
  {
    result = 1;
    if ( *(_BYTE *)(a1 + 416) )
      return *(unsigned __int8 *)(a1 + 72);
  }
  return result;
}
