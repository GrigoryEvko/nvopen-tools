// Function: sub_16EEC10
// Address: 0x16eec10
//
char *__fastcall sub_16EEC10(__int64 *a1, char *a2, char *a3, __int64 a4, __int64 a5)
{
  char *v5; // r13
  char *v7; // r14
  __int64 i; // r13
  __int64 v9; // rcx
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rbx
  char *v14; // rax
  __int64 v15; // r14
  __int64 *v16; // rbx
  char *v17; // r15
  char *v18; // rdx
  char *v19; // r12
  char *v20; // rax
  __int64 *v21; // rdi
  __int64 v22; // r8
  __int64 v23; // rcx
  __int64 v24; // rbx
  __int64 v25; // rbx
  char *j; // rax
  char *v27; // r13
  __int64 v28; // rax
  __int64 v29; // r13
  __int64 *v30; // rbx
  __int64 v31; // r15
  __int64 k; // r13
  __int64 v33; // rax
  __int64 v34; // r13
  char *v35; // rsi
  char *m; // rax
  char *v37; // r12
  __int64 v38; // r13
  char *v39; // rax
  char *v40; // rsi
  __int64 v41; // [rsp+0h] [rbp-60h]
  __int64 v42; // [rsp+8h] [rbp-58h]
  __int64 v43; // [rsp+10h] [rbp-50h]
  __int64 v44; // [rsp+10h] [rbp-50h]
  __int64 v45; // [rsp+18h] [rbp-48h]
  __int64 v46; // [rsp+18h] [rbp-48h]
  char *v47; // [rsp+18h] [rbp-48h]

  v5 = a2;
  if ( a4 < a5 )
  {
    v7 = a2;
    for ( i = a4; ; i = v12 )
    {
      v9 = *(_QWORD *)(*a1 + 8);
      v42 = 8 * i;
      v10 = *(_QWORD *)(v9 + 8 * i);
      v11 = (unsigned int)v10 & 0xF8000000;
      if ( v11 == 1476395008 )
        break;
      if ( v11 == 2013265920 )
      {
        v24 = i;
        do
        {
          v24 += v10 & 0x7FFFFFF;
          v10 = *(_QWORD *)(v9 + 8 * v24);
        }
        while ( (v10 & 0xF8000000) != 0x90000000LL );
        v25 = v24 + 1;
        v41 = i + 1;
        v45 = i;
        for ( j = sub_16EE780(a1, v7, a3, i, v25); ; j = sub_16EE780(a1, v7, v27 - 1, v45, v25) )
        {
          v27 = j;
          if ( a3 == sub_16EE780(a1, j, a3, v25, a5) )
            break;
        }
        v28 = *a1;
        v19 = v27;
        v29 = v45;
        v46 = v25;
        v30 = a1;
        v31 = v41;
        for ( k = v29 + (*(_QWORD *)(*(_QWORD *)(v28 + 8) + v42) & 0x7FFFFFFLL) - 1;
              v19 != sub_16EE780(v30, v7, v19, v31, k);
              k = v34 - ((*(_QWORD *)(v33 + 8 * v34) & 0xF8000000LL) == 2281701376LL) )
        {
          v31 = k + 2;
          v33 = *(_QWORD *)(*v30 + 8);
          v34 = k + 1 + (*(_QWORD *)(v33 + 8 * (k + 1)) & 0x7FFFFFFLL);
        }
        v23 = v31;
        a1 = v30;
        v12 = v46;
        v22 = k;
        goto LABEL_33;
      }
      if ( v11 == 1207959552 )
      {
        v43 = i + (v10 & 0x7FFFFFF);
        v12 = v43 + 1;
        for ( m = sub_16EE780(a1, v7, a3, i, v43 + 1); ; m = sub_16EE780(a1, v7, v37 - 1, i, v12) )
        {
          v37 = m;
          if ( a3 == sub_16EE780(a1, m, a3, v12, a5) )
            break;
        }
        v47 = v7;
        v38 = i + 1;
        while ( 1 )
        {
          v39 = sub_16EE780(a1, v7, v37, v38, v43);
          if ( !v39 )
            break;
          if ( v7 == v39 )
            goto LABEL_41;
          v47 = v7;
          v7 = v39;
        }
        v39 = v7;
        v7 = v47;
LABEL_41:
        v40 = v7;
        v7 = v37;
        sub_16EEC10(a1, v40, v39, v38, v43);
LABEL_11:
        if ( v12 >= a5 )
          return v7;
        continue;
      }
      v12 = i + 1;
      if ( v11 <= 0x48000000 )
      {
        if ( v11 == 671088640 || (v10 & 0xD8000000) == 0x10000000 )
          ++v7;
        goto LABEL_11;
      }
      if ( v11 == 1744830464 )
      {
        *(_QWORD *)(a1[2] + 16 * (v10 & 0x7FFFFFF)) = &v7[-a1[3]];
        goto LABEL_11;
      }
      if ( v11 <= 0x68000000 )
        goto LABEL_11;
      if ( v11 == 1879048192 )
      {
        *(_QWORD *)(a1[2] + 16 * (v10 & 0x7FFFFFF) + 8) = &v7[-a1[3]];
        goto LABEL_11;
      }
      if ( v12 >= a5 )
        return v7;
    }
    v44 = i + (v10 & 0x7FFFFFF);
    v14 = v7;
    v18 = a3;
    v15 = v44 + 1;
    v16 = a1;
    v17 = v14;
    while ( 1 )
    {
      v19 = sub_16EE780(v16, v17, v18, i, v15);
      if ( a3 == sub_16EE780(v16, v19, a3, v15, a5) )
        break;
      v18 = v19 - 1;
    }
    v20 = v17;
    a1 = v16;
    v21 = v16;
    v12 = v44 + 1;
    v7 = v20;
    if ( !sub_16EE780(v21, v20, v19, i + 1, v44) )
    {
      v7 = v19;
      goto LABEL_11;
    }
    v22 = v44;
    v23 = i + 1;
LABEL_33:
    v35 = v7;
    v7 = v19;
    sub_16EEC10(a1, v35, v19, v23, v22);
    goto LABEL_11;
  }
  return v5;
}
