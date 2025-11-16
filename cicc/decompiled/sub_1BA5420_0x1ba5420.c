// Function: sub_1BA5420
// Address: 0x1ba5420
//
__int64 __fastcall sub_1BA5420(
        __int64 a1,
        __m128 a2,
        __m128i a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 result; // rax
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rbx
  double v14; // xmm4_8
  double v15; // xmm5_8
  __int64 v16; // rdx
  int v17; // eax
  __int64 v18; // rsi
  int v19; // edx
  unsigned int v20; // eax
  int v21; // edi
  __int64 v22; // rcx

  result = sub_157F280(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 32LL));
  if ( v11 != result )
  {
    v12 = v11;
    v13 = result;
    do
    {
      if ( (unsigned __int8)sub_1BF28F0(*(_QWORD *)(a1 + 448), v13) )
      {
        sub_1B9D0E0(a1, v13, a2, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5, v14, v15, a8, a9);
      }
      else
      {
        v16 = *(_QWORD *)(a1 + 448);
        v17 = *(_DWORD *)(v16 + 96);
        if ( v17 )
        {
          v18 = *(_QWORD *)(v16 + 80);
          v19 = v17 - 1;
          v20 = (v17 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v21 = 1;
          v22 = *(_QWORD *)(v18 + 176LL * v20);
          if ( v22 == v13 )
          {
LABEL_12:
            sub_1BA4550(a1, v13, (__m128i)a2, a3, a4);
          }
          else
          {
            while ( v22 != -8 )
            {
              v20 = v19 & (v21 + v20);
              v22 = *(_QWORD *)(v18 + 176LL * v20);
              if ( v22 == v13 )
                goto LABEL_12;
              ++v21;
            }
          }
        }
      }
      if ( !v13 )
        BUG();
      result = *(_QWORD *)(v13 + 32);
      if ( !result )
        BUG();
      v13 = 0;
      if ( *(_BYTE *)(result - 8) == 77 )
        v13 = result - 24;
    }
    while ( v12 != v13 );
  }
  return result;
}
