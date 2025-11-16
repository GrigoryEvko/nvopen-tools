// Function: sub_11FCE30
// Address: 0x11fce30
//
__int64 *__fastcall sub_11FCE30(__int64 *a1, unsigned __int64 a2, __int64 a3)
{
  double v3; // xmm0_8
  double v6; // xmm1_8
  double v7; // xmm0_8

  v3 = 0.0;
  if ( a2 > a3 )
    a2 = a3;
  if ( a2 )
  {
    if ( (a2 & 0x8000000000000000LL) != 0LL )
    {
      v6 = log2((double)(int)(a2 & 1 | (a2 >> 1)) + (double)(int)(a2 & 1 | (a2 >> 1)));
      if ( a3 >= 0 )
        goto LABEL_7;
    }
    else
    {
      v6 = log2((double)(int)a2);
      if ( a3 >= 0 )
      {
LABEL_7:
        v7 = (double)(int)a3;
LABEL_8:
        v3 = v6 / log2(v7);
        goto LABEL_4;
      }
    }
    v7 = (double)(int)(a3 & 1 | ((unsigned __int64)a3 >> 1)) + (double)(int)(a3 & 1 | ((unsigned __int64)a3 >> 1));
    goto LABEL_8;
  }
LABEL_4:
  sub_11FCC80(a1, v3);
  return a1;
}
