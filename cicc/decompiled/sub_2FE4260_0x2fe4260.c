// Function: sub_2FE4260
// Address: 0x2fe4260
//
__int64 __fastcall sub_2FE4260(_BYTE *a1, size_t a2, signed __int64 *a3, unsigned __int8 *a4)
{
  _BYTE *v6; // rax
  signed __int64 v7; // rax
  size_t v8; // rax
  size_t v9; // rbx
  _BYTE *v11; // rax
  unsigned __int8 v12; // al

  if ( a2 && (v6 = memchr(a1, 58, a2)) != 0 )
  {
    v7 = v6 - a1;
    *a3 = v7;
    if ( v7 == -1 )
    {
      return 0;
    }
    else
    {
      v8 = v7 + 1;
      if ( v8 > a2 || (v9 = a2 - v8, a2 - v8 == -1) || (v11 = &a1[v8], v9 != 1) || (v12 = *v11 - 48, v12 > 9u) )
        sub_C64ED0("Invalid refinement step for -recip.", 1u);
      *a4 = v12;
      return 1;
    }
  }
  else
  {
    *a3 = -1;
    return 0;
  }
}
