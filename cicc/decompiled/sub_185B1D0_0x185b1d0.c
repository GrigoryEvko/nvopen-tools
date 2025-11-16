// Function: sub_185B1D0
// Address: 0x185b1d0
//
void __fastcall sub_185B1D0(
        __int64 a1,
        _BYTE *a2,
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
  __int64 v12; // rbx
  _QWORD *v13; // rax
  _QWORD *v14; // r12
  __int64 v15; // rax
  double v16; // xmm4_8
  double v17; // xmm5_8

  if ( a1 )
  {
    while ( 1 )
    {
      v12 = *(_QWORD *)(a1 + 8);
      v13 = sub_1648700(a1);
      v14 = v13;
      if ( *((_BYTE *)v13 + 16) > 0x17u && (v15 = sub_14DD210(v13, a2, a3)) != 0 )
      {
        sub_164D160((__int64)v14, v15, a4, a5, a6, a7, v16, v17, a10, a11);
        if ( !v12 )
        {
LABEL_13:
          if ( (unsigned __int8)sub_1AE9990(v14, a3) )
            sub_15F20C0(v14);
          return;
        }
        while ( v14 == sub_1648700(v12) )
        {
          v12 = *(_QWORD *)(v12 + 8);
          if ( !v12 )
            goto LABEL_13;
        }
        if ( (unsigned __int8)sub_1AE9990(v14, a3) )
          sub_15F20C0(v14);
      }
      else if ( !v12 )
      {
        return;
      }
      a1 = v12;
    }
  }
}
