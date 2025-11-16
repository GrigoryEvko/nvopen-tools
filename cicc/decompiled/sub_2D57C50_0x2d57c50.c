// Function: sub_2D57C50
// Address: 0x2d57c50
//
__int64 __fastcall sub_2D57C50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned __int64 v5; // rax
  __int64 v6; // rcx
  unsigned __int64 v7; // rax
  __int64 v9; // rdx

  v5 = *(unsigned __int8 *)(a2 + 8);
  if ( (unsigned __int8)v5 <= 0xCu && (v6 = 4143, _bittest64(&v6, v5)) )
  {
    v7 = *(unsigned __int8 *)(a3 + 8);
    if ( (unsigned __int8)v7 > 0xCu )
      goto LABEL_4;
  }
  else
  {
    a5 = 0;
    if ( (v5 & 0xFD) != 4 )
      return a5;
    v7 = *(unsigned __int8 *)(a3 + 8);
    if ( (unsigned __int8)v7 > 0xCu )
    {
LABEL_4:
      LOBYTE(a5) = (v7 & 0xFD) == 4;
      return a5;
    }
  }
  v9 = 4143;
  a5 = 1;
  if ( !_bittest64(&v9, v7) )
    goto LABEL_4;
  return 1;
}
