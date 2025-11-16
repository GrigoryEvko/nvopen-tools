// Function: sub_13E10E0
// Address: 0x13e10e0
//
unsigned __int8 *__fastcall sub_13E10E0(unsigned __int8 *a1, __int64 a2, char a3, char a4, _QWORD *a5)
{
  unsigned __int8 *result; // rax

  result = sub_13E0AE0(23, a1, a2, a5, 3);
  if ( !result )
    return (unsigned __int8 *)sub_13D0230(a1, a2, a3, a4);
  return result;
}
