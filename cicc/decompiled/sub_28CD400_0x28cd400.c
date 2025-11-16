// Function: sub_28CD400
// Address: 0x28cd400
//
__int64 __fastcall sub_28CD400(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        int a8,
        _QWORD *a9,
        _QWORD *a10,
        int a11,
        int a12,
        _QWORD *a13)
{
  _QWORD *v13; // rbx
  _QWORD *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 result; // rax

  v13 = a9;
  if ( a9 == a13 )
  {
    v16 = 0;
  }
  else
  {
    v14 = a9;
    v15 = 0;
    do
    {
      do
        ++v14;
      while ( v14 != a10 && (*v14 == -8192 || *v14 == -4096) );
      ++v15;
    }
    while ( a13 != v14 );
    v16 = v15;
  }
  v17 = *(unsigned int *)(a1 + 8);
  if ( v17 + v16 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v17 + v16, 8u, a5, a6);
    v17 = *(unsigned int *)(a1 + 8);
  }
  v18 = *(_QWORD *)a1 + 8 * v17;
  if ( a9 != a13 )
  {
    do
    {
      v18 += 8;
      *(_QWORD *)(v18 - 8) = *v13;
      do
        ++v13;
      while ( v13 != a10 && (*v13 == -8192 || *v13 == -4096) );
    }
    while ( v13 != a13 );
    v17 = *(unsigned int *)(a1 + 8);
  }
  result = v16 + v17;
  *(_DWORD *)(a1 + 8) = result;
  return result;
}
