// Function: sub_2DD3660
// Address: 0x2dd3660
//
__int64 __fastcall sub_2DD3660(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v4; // r13
  __int64 *v5; // r12
  __int64 v6; // r14
  int v7; // ebx
  __int64 v8; // rdi
  __int64 v9; // r12
  __int64 *v10; // r15
  int v11; // ebx
  __int64 v12; // rdi
  __int64 *v14; // [rsp+8h] [rbp-58h]
  __int64 v15; // [rsp+10h] [rbp-50h]
  int v16; // [rsp+18h] [rbp-48h]
  unsigned int v17; // [rsp+1Ch] [rbp-44h]
  __int64 v18; // [rsp+28h] [rbp-38h]

  v3 = a2 - a1;
  v15 = a1;
  v18 = 0xCCCCCCCCCCCCCCCDLL * (v3 >> 4);
  if ( v3 > 0 )
  {
    v14 = *(__int64 **)a3;
    v4 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
    v16 = *(_DWORD *)(a3 + 72);
    do
    {
      v5 = v14;
      v6 = v15 + 80 * (v18 >> 1);
      if ( v14 == (__int64 *)v4 )
      {
        v17 = 0;
      }
      else
      {
        v7 = 0;
        do
        {
          v8 = *v5++;
          v7 += sub_39FAC40(v8);
        }
        while ( (__int64 *)v4 != v5 );
        v17 = v16 * v7;
      }
      v9 = *(_QWORD *)v6 + 8LL * *(unsigned int *)(v6 + 8);
      if ( *(_QWORD *)v6 == v9 )
        goto LABEL_14;
      v10 = *(__int64 **)v6;
      v11 = 0;
      do
      {
        v12 = *v10++;
        v11 += sub_39FAC40(v12);
      }
      while ( (__int64 *)v9 != v10 );
      if ( *(_DWORD *)(v6 + 72) * v11 <= v17 )
      {
LABEL_14:
        v15 = v6 + 80;
        v18 = v18 - (v18 >> 1) - 1;
      }
      else
      {
        v18 >>= 1;
      }
    }
    while ( v18 > 0 );
  }
  return v15;
}
