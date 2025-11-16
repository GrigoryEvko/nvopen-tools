// Function: sub_1B76990
// Address: 0x1b76990
//
__int64 __fastcall sub_1B76990(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  double v12; // xmm4_8
  double v13; // xmm5_8
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // [rsp+0h] [rbp-30h] BYREF
  char v18; // [rsp+8h] [rbp-28h]

  if ( a3 )
  {
    sub_1B76840((__int64)&v17, *a2, a3, *(double *)a4.m128_u64, a5, a6);
    if ( v18 )
    {
      v15 = v17;
      *(_BYTE *)(a1 + 8) = 1;
      *(_QWORD *)a1 = v15;
      return a1;
    }
    else
    {
      if ( *(_BYTE *)(a3 + 1) == 1 )
      {
        v16 = sub_1B757D0((__int64)a2, a3, a4, a5, a6, a7, v12, v13, a10, a11);
        *(_BYTE *)(a1 + 8) = 1;
        *(_QWORD *)a1 = v16;
      }
      else
      {
        *(_BYTE *)(a1 + 8) = 0;
      }
      return a1;
    }
  }
  else
  {
    *(_BYTE *)(a1 + 8) = 1;
    *(_QWORD *)a1 = 0;
    return a1;
  }
}
