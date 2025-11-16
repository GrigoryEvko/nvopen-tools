// Function: sub_30D4FE0
// Address: 0x30d4fe0
//
__int64 __fastcall sub_30D4FE0(__int64 *a1, unsigned __int8 *a2, __int64 a3)
{
  int v5; // edx
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rbx
  int v10; // ebx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // rbx
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdx
  _DWORD *v21; // rax
  unsigned int v22; // eax
  unsigned int v23; // r13d
  __int64 v24; // r12
  __int64 v25; // rax
  __int64 result; // rax
  __int64 v27; // rax
  _QWORD *v28; // [rsp+10h] [rbp-60h]
  int v29; // [rsp+18h] [rbp-58h]
  __int64 v30; // [rsp+20h] [rbp-50h]
  __int64 v31; // [rsp+28h] [rbp-48h]
  _QWORD v32[8]; // [rsp+30h] [rbp-40h] BYREF

  v5 = *a2;
  if ( v5 == 40 )
  {
    v6 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v6 = 0;
    if ( v5 != 85 )
    {
      v6 = 64;
      if ( v5 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_10;
  v7 = sub_BD2BC0((__int64)a2);
  v9 = v7 + v8;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v9 >> 4) )
LABEL_30:
      BUG();
LABEL_10:
    v13 = 0;
    goto LABEL_11;
  }
  if ( !(unsigned int)((v9 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_10;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_30;
  v10 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v11 = sub_BD2BC0((__int64)a2);
  v13 = 32LL * (unsigned int)(*(_DWORD *)(v11 + v12 - 4) - v10);
LABEL_11:
  v14 = (32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v6 - v13) >> 5;
  if ( (_DWORD)v14 )
  {
    v15 = (unsigned int)v14;
    v16 = 0;
    v17 = 0;
    v31 = v15;
    v28 = a2 + 72;
    do
    {
      while ( !(unsigned __int8)sub_B49B80((__int64)a2, v17, 81) )
      {
        ++v17;
        v16 += (int)qword_5030168;
        if ( v31 == v17 )
          goto LABEL_19;
      }
      v30 = *(_QWORD *)(*(_QWORD *)&a2[32 * (v17 - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))] + 8LL);
      v18 = sub_A748A0(v28, v17);
      if ( !v18 )
      {
        v27 = *((_QWORD *)a2 - 4);
        if ( v27 )
        {
          if ( !*(_BYTE *)v27 && *(_QWORD *)(v27 + 24) == *((_QWORD *)a2 + 10) )
          {
            v32[0] = *(_QWORD *)(v27 + 120);
            v18 = sub_A748A0(v32, v17);
          }
        }
      }
      v19 = sub_9208B0(a3, v18);
      v32[1] = v20;
      v32[0] = v19;
      v29 = sub_CA1930(v32);
      v21 = sub_AE2980(a3, *(_DWORD *)(v30 + 8) >> 8);
      v22 = (unsigned int)(v21[1] + v29 - 1) / v21[1];
      if ( v22 > 8 )
        v22 = 8;
      ++v17;
      v16 += 2 * (unsigned int)qword_5030168 * v22;
    }
    while ( v31 != v17 );
  }
  else
  {
    v16 = 0;
  }
LABEL_19:
  v23 = qword_502FFA8;
  v24 = (int)qword_5030168 + v16;
  v25 = sub_B491C0((__int64)a2);
  result = v24 + (unsigned int)sub_DFE000(a1, v25, (__int64)a2, v23);
  if ( result > 0x7FFFFFFF )
    return 0x7FFFFFFF;
  return result;
}
