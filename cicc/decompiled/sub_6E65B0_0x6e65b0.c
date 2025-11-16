// Function: sub_6E65B0
// Address: 0x6e65b0
//
_QWORD *__fastcall sub_6E65B0(__int64 a1)
{
  _QWORD *result; // rax
  _DWORD *v2; // r13

  result = (_QWORD *)*(unsigned __int8 *)(a1 + 8);
  if ( (_BYTE)result )
  {
    if ( (_BYTE)result != 1 )
      sub_721090(a1);
    v2 = (_DWORD *)sub_6E1A20(a1);
    if ( (unsigned int)sub_6E5430() )
      sub_6851C0(0x922u, v2);
    return sub_6E6500(a1);
  }
  return result;
}
