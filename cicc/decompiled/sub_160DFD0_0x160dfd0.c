// Function: sub_160DFD0
// Address: 0x160dfd0
//
__int64 __fastcall sub_160DFD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v5; // r12
  __int64 *v6; // r14
  __int64 v7; // rsi
  __int64 *v8; // r12
  __int64 *v9; // r14
  __int64 v10; // rsi
  __int64 *v11; // r12
  __int64 *v12; // r14
  __int64 v13; // rsi
  __int64 *v14; // r12
  __int64 result; // rax
  __int64 *i; // r13
  __int64 v17; // rsi

  sub_16BD430(a3, *(unsigned __int8 *)(a2 + 168));
  sub_16BD4B0(a3, *(unsigned int *)(a2 + 16));
  v5 = *(__int64 **)(a2 + 8);
  v6 = &v5[*(unsigned int *)(a2 + 16)];
  while ( v6 != v5 )
  {
    v7 = *v5++;
    sub_16BD4C0(a3, v7);
  }
  sub_16BD4B0(a3, *(unsigned int *)(a2 + 96));
  v8 = *(__int64 **)(a2 + 88);
  v9 = &v8[*(unsigned int *)(a2 + 96)];
  while ( v9 != v8 )
  {
    v10 = *v8++;
    sub_16BD4C0(a3, v10);
  }
  sub_16BD4B0(a3, *(unsigned int *)(a2 + 128));
  v11 = *(__int64 **)(a2 + 120);
  v12 = &v11[*(unsigned int *)(a2 + 128)];
  while ( v12 != v11 )
  {
    v13 = *v11++;
    sub_16BD4C0(a3, v13);
  }
  sub_16BD4B0(a3, *(unsigned int *)(a2 + 160));
  v14 = *(__int64 **)(a2 + 152);
  result = *(unsigned int *)(a2 + 160);
  for ( i = &v14[result]; i != v14; result = sub_16BD4C0(a3, v17) )
    v17 = *v14++;
  return result;
}
