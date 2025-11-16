// Function: sub_1038EA0
// Address: 0x1038ea0
//
__int64 __fastcall sub_1038EA0(__int64 a1, __int64 a2, __int64 a3)
{
  float v3; // xmm1_4
  float v4; // xmm0_4
  float v5; // xmm0_4
  float v6; // xmm2_4
  __int64 result; // rax
  int v8; // eax

  if ( a2 < 0 )
    v3 = (float)(a2 & 1 | (unsigned int)((unsigned __int64)a2 >> 1))
       + (float)(a2 & 1 | (unsigned int)((unsigned __int64)a2 >> 1));
  else
    v3 = (float)(int)a2;
  if ( a1 < 0 )
    v4 = (float)(a1 & 1 | (unsigned int)((unsigned __int64)a1 >> 1))
       + (float)(a1 & 1 | (unsigned int)((unsigned __int64)a1 >> 1));
  else
    v4 = (float)(int)a1;
  v5 = (float)(v4 / v3) / 100.0;
  if ( *(float *)&qword_4F8F788[8] <= v5
    || (a3 < 0
      ? (v6 = (float)(a3 & 1 | (unsigned int)((unsigned __int64)a3 >> 1))
            + (float)(a3 & 1 | (unsigned int)((unsigned __int64)a3 >> 1)))
      : (v6 = (float)(int)a3),
        result = 2,
        (float)(v6 / v3) < (float)(1000 * LODWORD(qword_4F8F6A8[8]))) )
  {
    result = 1;
    if ( LOBYTE(qword_4F8F4E8[8]) )
    {
      v8 = qword_4F8F5C8[8];
      LOBYTE(v8) = v5 > (float)SLODWORD(qword_4F8F5C8[8]);
      return (unsigned int)(3 * v8 + 1);
    }
  }
  return result;
}
