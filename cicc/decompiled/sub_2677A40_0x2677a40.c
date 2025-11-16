// Function: sub_2677A40
// Address: 0x2677a40
//
__int64 __fastcall sub_2677A40(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        _QWORD *a8,
        int a9,
        int a10,
        _QWORD *a11)
{
  _QWORD *v11; // rbx
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rdx
  __int64 result; // rax

  v11 = a7;
  if ( a7 == a11 )
  {
    v14 = 0;
  }
  else
  {
    v12 = a7;
    v13 = 0;
    do
    {
      do
        ++v12;
      while ( v12 != a8 && *v12 >= 0xFFFFFFFFFFFFFFFELL );
      ++v13;
    }
    while ( a11 != v12 );
    v14 = v13;
  }
  v15 = *(unsigned int *)(a1 + 8);
  if ( v15 + v14 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v15 + v14, 8u, v15 + v14, a6);
    v15 = *(unsigned int *)(a1 + 8);
  }
  result = *(_QWORD *)a1 + 8 * v15;
  if ( a7 != a11 )
  {
    do
    {
      result += 8;
      *(_QWORD *)(result - 8) = *v11;
      do
        ++v11;
      while ( v11 != a8 && *v11 >= 0xFFFFFFFFFFFFFFFELL );
    }
    while ( v11 != a11 );
    LODWORD(v15) = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = v14 + v15;
  return result;
}
