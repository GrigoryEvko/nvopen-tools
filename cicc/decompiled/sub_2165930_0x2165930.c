// Function: sub_2165930
// Address: 0x2165930
//
__int64 __fastcall sub_2165930(__int64 a1)
{
  __int64 result; // rax
  _QWORD *v2; // rax

  result = sub_1F45DD0(a1);
  if ( (_DWORD)result && !byte_4FD1EC0 && !byte_4FD2080 )
  {
    v2 = (_QWORD *)sub_217DF60(*(_QWORD *)(a1 + 208));
    return sub_1F46490(a1, v2, 1, 1, 0);
  }
  return result;
}
