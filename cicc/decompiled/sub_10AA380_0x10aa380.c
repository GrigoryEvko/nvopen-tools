// Function: sub_10AA380
// Address: 0x10aa380
//
__int64 __fastcall sub_10AA380(__int64 a1, int a2, unsigned __int8 *a3)
{
  __int64 result; // rax
  _BYTE *v4; // rdx

  result = 0;
  if ( a2 + 29 == *a3 && *(_QWORD *)a1 == *((_QWORD *)a3 - 8) )
  {
    v4 = (_BYTE *)*((_QWORD *)a3 - 4);
    if ( *v4 <= 0x15u )
    {
      **(_QWORD **)(a1 + 8) = v4;
      return 1;
    }
  }
  return result;
}
