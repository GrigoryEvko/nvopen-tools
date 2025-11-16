// Function: sub_3037D10
// Address: 0x3037d10
//
__int64 __fastcall sub_3037D10(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned int *)(*(_QWORD *)(a1 + 537016) + 356LL);
  if ( (_DWORD)result == -1 )
    return 2 * (unsigned int)((*(_BYTE *)(*(_QWORD *)(a1 + 8) + 864LL) & 1) == 0);
  return result;
}
