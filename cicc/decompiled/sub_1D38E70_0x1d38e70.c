// Function: sub_1D38E70
// Address: 0x1d38e70
//
__int64 __fastcall sub_1D38E70(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 a4,
        __m128i a5,
        double a6,
        __m128i a7)
{
  unsigned int v8; // ebx
  __int64 v9; // rax
  unsigned int v10; // edx
  unsigned __int8 v11; // al

  v8 = a4;
  v9 = sub_1E0A0C0(*(_QWORD *)(a1 + 32));
  v10 = 8 * sub_15A9520(v9, 0);
  if ( v10 == 32 )
  {
    v11 = 5;
  }
  else if ( v10 > 0x20 )
  {
    v11 = 6;
    if ( v10 != 64 )
    {
      v11 = 0;
      if ( v10 == 128 )
        v11 = 7;
    }
  }
  else
  {
    v11 = 3;
    if ( v10 != 8 )
      v11 = 4 * (v10 == 16);
  }
  return sub_1D38BB0(a1, a2, a3, v11, 0, v8, a5, a6, a7, 0);
}
