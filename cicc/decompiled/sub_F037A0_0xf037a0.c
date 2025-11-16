// Function: sub_F037A0
// Address: 0xf037a0
//
__int64 __fastcall sub_F037A0(unsigned __int8 **a1, unsigned __int8 *a2, __int64 a3, __int64 a4, int a5)
{
  unsigned __int8 *v5; // rdi
  unsigned __int8 *v6; // r9
  char *v7; // r11
  __int64 result; // rax
  unsigned __int8 **v9; // r10
  int v10; // esi

  v5 = *a1;
  if ( a2 == v5 )
    return 1;
  v6 = a2;
  v7 = byte_3F88460;
  do
  {
    v10 = v7[*v5] + 1;
    if ( v10 > v6 - v5 )
      return 0;
    result = sub_F02F40(v5, v10, a3, a4, a5);
    if ( !(_BYTE)result )
      return result;
    v5 += v10;
    *v9 = v5;
  }
  while ( v5 != v6 );
  return 1;
}
