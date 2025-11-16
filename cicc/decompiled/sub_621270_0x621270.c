// Function: sub_621270
// Address: 0x621270
//
__int64 __fastcall sub_621270(unsigned __int16 *a1, __int16 *a2, int a3, _BOOL4 *a4)
{
  unsigned __int16 v4; // r10
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 result; // rax
  __int16 v10; // r10
  _BOOL4 v11; // edx

  v4 = *a1;
  v7 = 7;
  v8 = 0;
  do
  {
    result = v8 + (unsigned __int16)a2[v7] + (unsigned __int64)a1[v7];
    v8 = 0;
    if ( result > 0xFFFF )
    {
      result -= 0x10000;
      v8 = 1;
    }
    a1[v7--] = result;
  }
  while ( v7 != -1 );
  if ( a3 )
  {
    result = (unsigned __int16)*a2 >> 15;
    v10 = v4 >> 15;
    v11 = 0;
    if ( (_BYTE)v10 == *a2 < 0 )
    {
      result = *a1 >> 15;
      v11 = (_BYTE)v10 != ((*a1 & 0x8000u) != 0);
    }
    *a4 = v11;
  }
  else
  {
    *a4 = v8;
  }
  return result;
}
