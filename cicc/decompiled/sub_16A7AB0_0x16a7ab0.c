// Function: sub_16A7AB0
// Address: 0x16a7ab0
//
__int64 __fastcall sub_16A7AB0(_QWORD *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  _QWORD *v6; // r13
  __int64 v7; // rbx
  unsigned int v8; // r12d
  unsigned __int64 v9; // rdx
  unsigned int v10; // r9d
  __int64 v11; // rdi

  v6 = a1;
  sub_16A7020(a1, 0, a4);
  if ( a4 )
  {
    v7 = 0;
    v8 = 0;
    do
    {
      v9 = *(_QWORD *)(a3 + 8 * v7);
      v10 = a4 - v7;
      v11 = (__int64)v6;
      ++v7;
      ++v6;
      v8 |= sub_16A7890(v11, a2, v9, 0, a4, v10, 1);
    }
    while ( v7 != a4 );
  }
  else
  {
    return 0;
  }
  return v8;
}
