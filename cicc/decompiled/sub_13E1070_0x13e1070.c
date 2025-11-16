// Function: sub_13E1070
// Address: 0x13e1070
//
unsigned __int8 *__fastcall sub_13E1070(unsigned __int8 *a1, _BYTE *a2, char a3, _QWORD *a4)
{
  unsigned __int8 *result; // rax

  result = sub_13E0EE0(24, a1, (__int64)a2, a3, a4, 3);
  if ( !result )
    return (unsigned __int8 *)sub_13D7050((__int64)a1, a2, a4);
  return result;
}
