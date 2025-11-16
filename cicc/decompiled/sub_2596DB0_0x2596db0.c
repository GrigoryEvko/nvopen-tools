// Function: sub_2596DB0
// Address: 0x2596db0
//
__int64 __fastcall sub_2596DB0(__int64 a1, __int64 a2, __m128i *a3, int a4, _BYTE *a5, char a6, __int64 *a7)
{
  unsigned int v9; // r12d
  __int64 v11; // rax

  *a5 = 0;
  v9 = sub_2553BE0(a1, a3, 22, a6);
  if ( (_BYTE)v9 )
  {
    *a5 = 1;
  }
  else if ( a2 )
  {
    v11 = sub_2596900(a1, a3->m128i_i64[0], a3->m128i_i64[1], a2, a4, 0, 1);
    if ( a7 )
      *a7 = v11;
    if ( v11 )
    {
      v9 = *(unsigned __int8 *)(v11 + 97);
      if ( (_BYTE)v9 )
        *a5 = *(_BYTE *)(v11 + 96);
    }
  }
  return v9;
}
