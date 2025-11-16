// Function: sub_133DC20
// Address: 0x133dc20
//
__int64 __fastcall sub_133DC20(__int64 *a1, unsigned __int64 a2, unsigned __int64 a3)
{
  __int64 v5; // rsi
  unsigned __int64 v6; // rcx
  __int64 v7; // rax
  __int64 i; // rdx
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned __int64 v13; // r10
  unsigned __int64 j; // r9
  unsigned __int64 v15; // r10
  __int64 v16; // r13
  unsigned __int64 v17; // r8
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // r14
  unsigned __int64 v22; // rcx
  __int64 k; // rax
  __int64 v24; // r15
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // rax
  __int64 m; // r15
  __int64 v29; // r15
  _QWORD *v30; // rax
  __int64 v32; // [rsp-50h] [rbp-50h]
  __int64 v33; // [rsp-48h] [rbp-48h]
  __int64 v34; // [rsp-40h] [rbp-40h]

  if ( a1[15] <= 0 )
    return -1;
  v32 = sub_130B0E0((__int64)(a1 + 16));
  if ( !a2 )
  {
    v30 = a1 + 22;
    while ( !*v30 )
    {
      if ( a1 + 222 == ++v30 )
        return -1;
    }
    return 200 * v32;
  }
  if ( a2 <= a3 )
    return 200 * v32;
  v33 = a1[22];
  v5 = 20;
  v6 = 165 * a1[23] + 20 * v33;
  v7 = 553;
  for ( i = 2; ; v5 = qword_42878A0[i - 2] )
  {
    v9 = a1[i++ + 22] * (v7 - v5);
    v6 += v9;
    if ( i == 200 )
      break;
    v7 = qword_42878A0[i];
  }
  v10 = v6 >> 24;
  if ( a3 >= v10 )
  {
    v11 = a1[22];
    v12 = 20;
    v13 = 0;
    for ( j = 0; ; v12 = qword_42878A0[j] )
    {
      ++j;
      v13 += v12 * v11;
      if ( j == 200 )
        break;
      v11 = a1[j + 22];
    }
    v15 = v13 >> 24;
    v16 = 2;
    if ( a3 <= v15 )
    {
LABEL_12:
      v17 = v10 + a3;
      while ( 1 )
      {
        v18 = j + v16;
        if ( v15 <= v17 || v16 + 2 >= j )
          return (v32 * v18) >> 1;
        v19 = v18 >> 1;
        if ( !v19 )
          break;
        v20 = v33;
        v21 = 20;
        v22 = 0;
        for ( k = 0; ; v21 = qword_42878A0[k] )
        {
          ++k;
          v22 += v21 * v20;
          if ( v19 == k )
            break;
          v20 = a1[k + 22];
        }
        if ( v19 != 200 )
        {
          v24 = a1[v19 + 22];
          v25 = qword_42878A0[v19];
          v26 = v19;
          goto LABEL_21;
        }
LABEL_24:
        v10 = v22 >> 24;
        if ( a3 >= v10 )
        {
          v16 = v19;
          goto LABEL_12;
        }
        v15 = v10;
        j = v19;
      }
      v24 = v33;
      v22 = 0;
      v26 = 0;
      v25 = 20;
LABEL_21:
      v34 = v25;
      v27 = v24;
      for ( m = v34; ; m = qword_42878A0[v26] )
      {
        v29 = m - qword_42878A0[v26++ - v19];
        v22 += v27 * v29;
        if ( v26 == 200 )
          break;
        v27 = a1[v26 + 22];
      }
      goto LABEL_24;
    }
    return 200 * v32;
  }
  return 2 * v32;
}
