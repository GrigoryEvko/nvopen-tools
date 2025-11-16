// Function: sub_19E7570
// Address: 0x19e7570
//
void __fastcall sub_19E7570(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        int a5,
        int a6,
        int a7,
        int a8,
        _QWORD *a9,
        _QWORD *a10,
        int a11,
        int a12,
        _QWORD *a13)
{
  _QWORD *v13; // rbx
  __int64 v14; // rsi
  _QWORD *v15; // rax
  unsigned __int64 v16; // rcx
  int v17; // r13d
  __int64 v18; // rdx

  v13 = a9;
  v14 = *(unsigned int *)(a1 + 8);
  if ( a9 == a13 )
  {
    v17 = 0;
  }
  else
  {
    v15 = a9;
    v16 = 0;
    do
    {
      do
        ++v15;
      while ( v15 != a10 && (*v15 == -16 || *v15 == -8) );
      ++v16;
    }
    while ( a13 != v15 );
    v17 = v16;
    if ( (unsigned __int64)*(unsigned int *)(a1 + 12) - v14 < v16 )
    {
      sub_16CD150(a1, (const void *)(a1 + 16), v16 + v14, 8, a5, a6);
      v14 = *(unsigned int *)(a1 + 8);
    }
    v18 = *(_QWORD *)a1 + 8 * v14;
    do
    {
      v18 += 8;
      *(_QWORD *)(v18 - 8) = *v13;
      do
        ++v13;
      while ( v13 != a10 && (*v13 == -16 || *v13 == -8) );
    }
    while ( v13 != a13 );
    LODWORD(v14) = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = v14 + v17;
}
