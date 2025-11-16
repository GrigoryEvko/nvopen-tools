// Function: sub_2640D80
// Address: 0x2640d80
//
__int64 __fastcall sub_2640D80(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        int a8,
        _DWORD *a9,
        _DWORD *a10,
        int a11,
        int a12,
        _DWORD *a13)
{
  _DWORD *v13; // rbx
  _DWORD *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r13
  __int64 result; // rax

  v13 = a9;
  if ( a13 == a9 )
  {
    *a1 = 0;
    result = 0;
    a1[2] = 0;
  }
  else
  {
    v14 = a9;
    v15 = 0;
    do
    {
      do
        ++v14;
      while ( v14 != a10 && *v14 > 0xFFFFFFFD );
      ++v15;
    }
    while ( a13 != v14 );
    if ( v15 > 0x1FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v16 = 4 * v15;
    result = sub_22077B0(4 * v15);
    *a1 = result;
    a1[2] = result + v16;
    do
    {
      result += 4;
      *(_DWORD *)(result - 4) = *v13;
      do
        ++v13;
      while ( v13 != a10 && *v13 > 0xFFFFFFFD );
    }
    while ( a13 != v13 );
  }
  a1[1] = result;
  return result;
}
