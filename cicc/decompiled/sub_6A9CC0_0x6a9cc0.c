// Function: sub_6A9CC0
// Address: 0x6a9cc0
//
unsigned int *__fastcall sub_6A9CC0(unsigned __int8 a1, _QWORD *a2, __int64 a3)
{
  __int64 v4; // rax
  unsigned int *result; // rax

  v4 = sub_68AFD0(a1);
  sub_6A9320(a2, a1, v4, 1, 1u, 1, a3);
  result = &dword_4D044B0;
  if ( !dword_4D044B0 )
    return (unsigned int *)sub_6E6840(a3);
  return result;
}
