// Function: sub_8EC360
// Address: 0x8ec360
//
unsigned __int8 *__fastcall sub_8EC360(unsigned __int8 *a1, _DWORD *a2, __int64 a3)
{
  unsigned __int8 *v3; // r10
  __int64 v4; // r9
  unsigned __int8 *result; // rax
  _DWORD v6[3]; // [rsp+Ch] [rbp-14h] BYREF

  v3 = a1;
  v4 = a3;
  if ( !*(_QWORD *)(a3 + 32) )
    sub_8E5790(" ::", a3);
  if ( *v3 == 83 && v3[1] == 116 )
  {
    if ( !*(_QWORD *)(v4 + 32) )
      sub_8E5790("std::", v4);
    v3 += 2;
  }
  result = sub_8EBEA0(v3, v6, v4);
  *a2 = v6[0];
  return result;
}
