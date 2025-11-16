// Function: sub_16ED430
// Address: 0x16ed430
//
char *__fastcall sub_16ED430(_QWORD *a1, char *a2, char *a3, __int64 a4, __int64 a5)
{
  char *v5; // r13
  char *v7; // r14
  __int64 i; // r13
  __int64 v9; // r11
  __int64 v10; // r9
  unsigned __int64 v11; // rax
  __int64 v12; // rbx
  _QWORD *v14; // rax
  __int64 v15; // r12
  __int64 v16; // rbx
  char *v17; // rdx
  char *v18; // r15
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // rbx
  char *j; // rax
  char *v24; // r13
  __int64 v25; // r11
  char *v26; // r15
  __int64 v27; // r13
  __int64 v28; // rbx
  __int64 v29; // r13
  __int64 v30; // r12
  char *k; // rax
  __int64 v32; // r13
  __int64 v33; // rcx
  char *v34; // rsi
  char *m; // rax
  char *v36; // r15
  __int64 v37; // r13
  char *v38; // rax
  char *v39; // rsi
  int v40; // [rsp+0h] [rbp-60h]
  __int64 v41; // [rsp+8h] [rbp-58h]
  __int64 v42; // [rsp+8h] [rbp-58h]
  __int64 v43; // [rsp+10h] [rbp-50h]
  __int64 v44; // [rsp+10h] [rbp-50h]
  __int64 v45; // [rsp+10h] [rbp-50h]
  __int64 v46; // [rsp+10h] [rbp-50h]
  __int64 v47; // [rsp+18h] [rbp-48h]
  __int64 v48; // [rsp+18h] [rbp-48h]
  char *v49; // [rsp+18h] [rbp-48h]

  v5 = a2;
  if ( a4 < a5 )
  {
    v7 = a2;
    for ( i = a4; ; i = v12 )
    {
      v9 = *(_QWORD *)(*a1 + 8LL);
      v10 = *(_QWORD *)(v9 + 8 * i);
      v11 = (unsigned int)v10 & 0xF8000000;
      if ( v11 == 1476395008 )
        break;
      if ( v11 == 2013265920 )
      {
        v20 = *(_QWORD *)(v9 + 8 * i);
        v21 = i;
        do
        {
          v21 += v20 & 0x7FFFFFF;
          v20 = *(_QWORD *)(v9 + 8 * v21);
        }
        while ( (v20 & 0xF8000000) != 0x90000000LL );
        v22 = v21 + 1;
        v41 = i + 1;
        v43 = *(_QWORD *)(*a1 + 8LL);
        v40 = v10;
        v47 = i;
        for ( j = sub_16ECFD0((__int64)a1, v7, a3, i, v22); ; j = sub_16ECFD0((__int64)a1, v7, v24 - 1, v47, v22) )
        {
          v24 = j;
          if ( a3 == sub_16ECFD0((__int64)a1, j, a3, v22, a5) )
            break;
        }
        v25 = v43;
        v26 = v24;
        v44 = v22;
        v27 = v47;
        v48 = (__int64)a1;
        v28 = v25;
        v29 = v27 + (v40 & 0x7FFFFFF) - 1;
        v30 = v41;
        for ( k = sub_16ECFD0(v48, v7, v26, v41, v29); v26 != k; k = sub_16ECFD0(v48, v7, v26, v30, v29) )
        {
          v30 = v29 + 2;
          v32 = (*(_QWORD *)(v28 + 8 * (v29 + 1)) & 0x7FFFFFFLL) + v29 + 1;
          v29 = v32 - ((*(_QWORD *)(v28 + 8 * v32) & 0xF8000000LL) == 2281701376LL);
        }
        v33 = v30;
        a1 = (_QWORD *)v48;
        v34 = v7;
        v12 = v44;
        v7 = v26;
        sub_16ED430(v48, v34, v26, v33, v29);
        goto LABEL_11;
      }
      if ( v11 == 1207959552 )
      {
        v45 = (v10 & 0x7FFFFFF) + i;
        v12 = v45 + 1;
        for ( m = sub_16ECFD0((__int64)a1, v7, a3, i, v45 + 1); ; m = sub_16ECFD0((__int64)a1, v7, v36 - 1, i, v12) )
        {
          v36 = m;
          if ( a3 == sub_16ECFD0((__int64)a1, m, a3, v12, a5) )
            break;
        }
        v49 = v7;
        v37 = i + 1;
        while ( 1 )
        {
          v38 = sub_16ECFD0((__int64)a1, v7, v36, v37, v45);
          if ( !v38 )
            break;
          if ( v7 == v38 )
            goto LABEL_41;
          v49 = v7;
          v7 = v38;
        }
        v38 = v7;
        v7 = v49;
LABEL_41:
        v39 = v7;
        v7 = v36;
        sub_16ED430(a1, v39, v38, v37, v45);
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
    v46 = (v10 & 0x7FFFFFF) + i;
    v42 = i + 1;
    v14 = a1;
    v17 = a3;
    v15 = v46 + 1;
    v16 = (__int64)v14;
    while ( 1 )
    {
      v18 = sub_16ECFD0(v16, v7, v17, i, v15);
      if ( a3 == sub_16ECFD0(v16, v18, a3, v15, a5) )
        break;
      v17 = v18 - 1;
    }
    v19 = v16;
    v12 = v46 + 1;
    a1 = (_QWORD *)v19;
    if ( sub_16ECFD0(v19, v7, v18, v42, v46) )
      sub_16ED430(v19, v7, v18, v42, v46);
    v7 = v18;
    goto LABEL_11;
  }
  return v5;
}
