// Function: sub_29B8240
// Address: 0x29b8240
//
double __fastcall sub_29B8240(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, char a5)
{
  unsigned __int64 v5; // rsi
  unsigned __int64 v6; // rsi
  __int64 *v7; // rax
  double result; // xmm0_8
  double v9; // xmm2_8
  int v10; // eax
  double v11; // xmm1_8
  double v12; // xmm0_8
  double v13; // xmm1_8
  unsigned __int64 v14; // rdx
  __int64 *v15; // rax
  __int64 *v16; // rax

  v5 = a1 + a2;
  if ( v5 == a3 )
  {
    v16 = &qword_50084E0;
    if ( !a5 )
      v16 = &qword_5008400;
    if ( a4 < 0 )
      return ((double)(int)(a4 & 1 | ((unsigned __int64)a4 >> 1)) + (double)(int)(a4 & 1 | ((unsigned __int64)a4 >> 1)))
           * *((double *)v16 + 17);
    else
      return (double)(int)a4 * *((double *)v16 + 17);
  }
  else if ( v5 < a3 )
  {
    v14 = a3 - v5;
    v15 = &qword_5008860;
    result = 0.0;
    if ( !a5 )
      v15 = &qword_5008780;
    v9 = *((double *)v15 + 17);
    v10 = qword_50083A8;
    if ( (unsigned int)qword_50083A8 >= v14 )
    {
      v11 = (double)(int)v14;
      goto LABEL_7;
    }
  }
  else
  {
    v6 = v5 - a3;
    v7 = &qword_50086A0;
    result = 0.0;
    if ( !a5 )
      v7 = &qword_50085C0;
    v9 = *((double *)v7 + 17);
    v10 = qword_50082C8;
    if ( (unsigned int)qword_50082C8 >= v6 )
    {
      v11 = (double)(int)v6;
LABEL_7:
      v12 = (1.0 - v11 / (double)v10) * v9;
      if ( a4 < 0 )
        v13 = (double)(int)(a4 & 1 | ((unsigned __int64)a4 >> 1)) + (double)(int)(a4 & 1 | ((unsigned __int64)a4 >> 1));
      else
        v13 = (double)(int)a4;
      return v12 * v13;
    }
  }
  return result;
}
