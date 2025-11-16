// Function: sub_2DD3790
// Address: 0x2dd3790
//
__int64 __fastcall sub_2DD3790(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v4; // r13
  __int64 v5; // r14
  __int64 v6; // r12
  __int64 *v7; // r15
  int v8; // ebx
  __int64 v9; // rdi
  unsigned int v10; // r15d
  __int64 *v11; // r12
  int v12; // ebx
  __int64 v13; // rdi
  __int64 *v16; // [rsp+10h] [rbp-50h]
  __int64 v17; // [rsp+18h] [rbp-48h]
  __int64 v18; // [rsp+28h] [rbp-38h]

  v3 = a2 - a1;
  v17 = a1;
  v18 = 0xCCCCCCCCCCCCCCCDLL * (v3 >> 4);
  if ( v3 > 0 )
  {
    v16 = *(__int64 **)a3;
    v4 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
    do
    {
      v5 = v17 + 80 * (v18 >> 1);
      v6 = *(_QWORD *)v5 + 8LL * *(unsigned int *)(v5 + 8);
      if ( *(_QWORD *)v5 == v6 )
      {
        v8 = 0;
      }
      else
      {
        v7 = *(__int64 **)v5;
        v8 = 0;
        do
        {
          v9 = *v7++;
          v8 += sub_39FAC40(v9);
        }
        while ( (__int64 *)v6 != v7 );
      }
      v10 = *(_DWORD *)(v5 + 72) * v8;
      if ( v16 == (__int64 *)v4 )
        goto LABEL_13;
      v11 = v16;
      v12 = 0;
      do
      {
        v13 = *v11++;
        v12 += sub_39FAC40(v13);
      }
      while ( (__int64 *)v4 != v11 );
      if ( v10 >= *(_DWORD *)(a3 + 72) * v12 )
      {
LABEL_13:
        v18 >>= 1;
      }
      else
      {
        v17 = v5 + 80;
        v18 = v18 - (v18 >> 1) - 1;
      }
    }
    while ( v18 > 0 );
  }
  return v17;
}
