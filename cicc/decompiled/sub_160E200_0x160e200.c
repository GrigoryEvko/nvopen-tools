// Function: sub_160E200
// Address: 0x160e200
//
__int64 __fastcall sub_160E200(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v8; // rbx
  __int64 *v9; // r15
  __int64 v10; // rsi
  __int64 *v11; // rbx
  __int64 *v12; // r15
  __int64 v13; // rsi
  __int64 *v14; // rbx
  __int64 *v15; // r15
  __int64 v16; // rsi
  __int64 *v17; // rbx
  __int64 *v18; // r14
  __int64 v19; // rsi

  sub_16BD430(a5, *(unsigned __int8 *)(a2 + 168));
  sub_16BD4B0(a5, *(unsigned int *)(a2 + 16));
  v8 = *(__int64 **)(a2 + 8);
  v9 = &v8[*(unsigned int *)(a2 + 16)];
  while ( v9 != v8 )
  {
    v10 = *v8++;
    sub_16BD4C0(a5, v10);
  }
  sub_16BD4B0(a5, *(unsigned int *)(a2 + 96));
  v11 = *(__int64 **)(a2 + 88);
  v12 = &v11[*(unsigned int *)(a2 + 96)];
  while ( v12 != v11 )
  {
    v13 = *v11++;
    sub_16BD4C0(a5, v13);
  }
  sub_16BD4B0(a5, *(unsigned int *)(a2 + 128));
  v14 = *(__int64 **)(a2 + 120);
  v15 = &v14[*(unsigned int *)(a2 + 128)];
  while ( v15 != v14 )
  {
    v16 = *v14++;
    sub_16BD4C0(a5, v16);
  }
  sub_16BD4B0(a5, *(unsigned int *)(a2 + 160));
  v17 = *(__int64 **)(a2 + 152);
  v18 = &v17[*(unsigned int *)(a2 + 160)];
  while ( v18 != v17 )
  {
    v19 = *v17++;
    sub_16BD4C0(a5, v19);
  }
  return sub_16BD750(a5, a3);
}
