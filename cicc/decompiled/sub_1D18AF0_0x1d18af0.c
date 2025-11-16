// Function: sub_1D18AF0
// Address: 0x1d18af0
//
__int64 __fastcall sub_1D18AF0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rbx
  __int64 *v4; // r14
  __int64 v5; // rsi

  sub_16BD430(a2, *(unsigned __int16 *)(a1 + 24));
  sub_16BD4C0(a2, *(_QWORD *)(a1 + 40));
  v3 = *(__int64 **)(a1 + 32);
  v4 = &v3[5 * *(unsigned int *)(a1 + 56)];
  while ( v4 != v3 )
  {
    v5 = *v3;
    v3 += 5;
    sub_16BD4C0(a2, v5);
    sub_16BD430(a2, *((_DWORD *)v3 - 8));
  }
  return sub_1D14B60(a2, a1);
}
