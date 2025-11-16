// Function: sub_1A1DA80
// Address: 0x1a1da80
//
__int64 __fastcall sub_1A1DA80(
        __int64 *a1,
        _BYTE *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        int a8,
        __int16 a9)
{
  _BYTE *v9; // r13
  unsigned int v10; // eax
  __int64 **v11; // r15
  unsigned __int64 v13; // r12
  __int64 *v14; // rdi
  unsigned int v15; // ebx
  char v16; // al
  __int64 *v18; // rdx
  __m128i v19; // [rsp+10h] [rbp-50h] BYREF
  __int16 v20; // [rsp+20h] [rbp-40h]

  v9 = a2;
  v10 = *(_DWORD *)(a3 + 8);
  if ( v10 )
  {
    v11 = *(__int64 ***)a3;
    v13 = v10;
    if ( v10 == 1 )
    {
      v14 = *v11;
      v15 = *((_DWORD *)*v11 + 8);
      if ( v15 <= 0x40 )
      {
        if ( !v14[3] )
          return (__int64)v9;
      }
      else if ( v15 == (unsigned int)sub_16A57B0((__int64)(v14 + 3)) )
      {
        return (__int64)v9;
      }
    }
    v16 = a9;
    if ( (_BYTE)a9 )
    {
      if ( (_BYTE)a9 == 1 )
      {
        v19.m128i_i64[0] = (__int64)"sroa_idx";
        v20 = 259;
      }
      else
      {
        v18 = a7;
        if ( HIBYTE(a9) != 1 )
        {
          v18 = (__int64 *)&a7;
          v16 = 2;
        }
        v19.m128i_i64[0] = (__int64)v18;
        v19.m128i_i64[1] = (__int64)"sroa_idx";
        LOBYTE(v20) = v16;
        HIBYTE(v20) = 3;
      }
    }
    else
    {
      v20 = 256;
    }
    return sub_1A1D720(a1, a2, v11, v13, &v19);
  }
  return (__int64)v9;
}
