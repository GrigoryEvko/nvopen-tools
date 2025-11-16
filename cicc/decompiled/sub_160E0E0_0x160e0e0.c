// Function: sub_160E0E0
// Address: 0x160e0e0
//
__int64 __fastcall sub_160E0E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v5; // rbx
  __int64 *v6; // r14
  __int64 v7; // rsi
  __int64 *v8; // rbx
  __int64 *v9; // r14
  __int64 v10; // rsi
  __int64 *v11; // rbx
  __int64 *v12; // r14
  __int64 v13; // rsi
  __int64 *v14; // rbx
  __int64 *v15; // r13
  __int64 v16; // rsi

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
  v15 = &v14[*(unsigned int *)(a2 + 160)];
  while ( v15 != v14 )
  {
    v16 = *v14++;
    sub_16BD4C0(a3, v16);
  }
  return sub_16BDDB0(a3);
}
