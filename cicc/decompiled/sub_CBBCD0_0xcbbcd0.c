// Function: sub_CBBCD0
// Address: 0xcbbcd0
//
char *__fastcall sub_CBBCD0(__int64 *a1, char *a2, char *a3, __int64 a4, __int64 a5)
{
  char *v5; // r13
  __int64 v7; // r15
  char *v8; // r14
  __int64 v9; // r13
  __int64 v10; // r11
  __int64 v11; // r10
  unsigned __int64 v12; // rax
  __int64 v13; // rbx
  char *m; // rdx
  __int64 v16; // r8
  __int64 v17; // rcx
  char *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // rbx
  char *i; // rax
  char *v23; // r14
  __int64 v24; // r11
  char *v25; // r13
  char *v26; // r14
  __int64 v27; // rbx
  __int64 v28; // r15
  __int64 v29; // r12
  char *j; // rax
  __int64 v31; // r15
  __int64 v32; // rcx
  char *v33; // rsi
  char *k; // rax
  char *v35; // rdx
  __int64 v36; // r13
  char *v37; // r12
  __int64 v38; // r15
  char *v39; // rax
  char *v40; // r10
  __int64 v41; // [rsp+0h] [rbp-60h]
  __int64 v42; // [rsp+0h] [rbp-60h]
  int v43; // [rsp+8h] [rbp-58h]
  __int64 v44; // [rsp+8h] [rbp-58h]
  __int64 v45; // [rsp+8h] [rbp-58h]
  __int64 v46; // [rsp+10h] [rbp-50h]
  __int64 v47; // [rsp+10h] [rbp-50h]
  char *v48; // [rsp+18h] [rbp-48h]
  char *v49; // [rsp+18h] [rbp-48h]
  __int64 v50; // [rsp+18h] [rbp-48h]
  unsigned __int64 v51; // [rsp+18h] [rbp-48h]
  char *v52; // [rsp+18h] [rbp-48h]

  v5 = a2;
  if ( a4 < a5 )
  {
    v7 = a4;
    v8 = a2;
    while ( 1 )
    {
      v10 = *(_QWORD *)(*a1 + 8);
      v11 = *(_QWORD *)(v10 + 8 * v7);
      v12 = (unsigned int)v11 & 0xF8000000;
      if ( v12 == 1476395008 )
        break;
      if ( v12 == 2013265920 )
      {
        v19 = *(_QWORD *)(v10 + 8 * v7);
        v20 = v7;
        do
        {
          v20 += v19 & 0x7FFFFFF;
          v19 = *(_QWORD *)(v10 + 8 * v20);
        }
        while ( (v19 & 0xF8000000) != 0x90000000LL );
        v21 = v20 + 1;
        v41 = v7 + 1;
        v46 = *(_QWORD *)(*a1 + 8);
        v43 = v11;
        v49 = v8;
        for ( i = sub_CBB820((__int64)a1, v8, a3, v7, v21); ; i = sub_CBB820((__int64)a1, v49, v23 - 1, v7, v21) )
        {
          v23 = i;
          if ( a3 == sub_CBB820((__int64)a1, i, a3, v21, a5) )
            break;
        }
        v24 = v46;
        v25 = v23;
        v47 = v21;
        v26 = v49;
        v50 = (__int64)a1;
        v27 = v24;
        v28 = v7 + (v43 & 0x7FFFFFF) - 1;
        v29 = v41;
        for ( j = sub_CBB820(v50, v26, v25, v41, v28); v25 != j; j = sub_CBB820(v50, v26, v25, v29, v28) )
        {
          v29 = v28 + 2;
          v31 = (*(_QWORD *)(v27 + 8 * (v28 + 1)) & 0x7FFFFFFLL) + v28 + 1;
          v28 = v31 - ((*(_QWORD *)(v27 + 8 * v31) & 0xF8000000LL) == 2281701376LL);
        }
        v32 = v29;
        a1 = (__int64 *)v50;
        v33 = v26;
        v13 = v47;
        v8 = v25;
        sub_CBBCD0(v50, v33, v25, v32, v28);
        goto LABEL_11;
      }
      if ( v12 == 1207959552 )
      {
        v42 = *a1;
        v44 = (v11 & 0x7FFFFFF) + v7;
        v13 = v44 + 1;
        for ( k = sub_CBB820((__int64)a1, v8, a3, v7, v44 + 1); ; k = sub_CBB820((__int64)a1, v8, v35, v7, v13) )
        {
          v51 = (unsigned __int64)k;
          if ( a3 == sub_CBB820((__int64)a1, k, a3, v13, a5) )
            break;
          v35 = sub_CBAF40(v42, v8, v51, v13, a5);
        }
        v36 = (__int64)a1;
        v37 = (char *)v51;
        v52 = v8;
        v38 = v7 + 1;
        while ( 1 )
        {
          v39 = sub_CBB820(v36, v8, v37, v38, v44);
          if ( !v39 )
            break;
          if ( v8 == v39 )
          {
            v40 = v37;
            a1 = (__int64 *)v36;
            goto LABEL_42;
          }
          v52 = v8;
          v8 = v39;
        }
        v40 = v37;
        a1 = (__int64 *)v36;
        v39 = v8;
        v8 = v52;
LABEL_42:
        v48 = v40;
        v16 = v44;
        v17 = v38;
        v18 = v39;
        goto LABEL_21;
      }
      v13 = v7 + 1;
      if ( v12 <= 0x48000000 )
      {
        if ( v12 == 671088640 || (v11 & 0xD8000000) == 0x10000000 )
          ++v8;
LABEL_11:
        if ( v13 >= a5 )
          return v8;
        goto LABEL_12;
      }
      if ( v12 == 1744830464 )
      {
        *(_QWORD *)(a1[2] + 16 * (v11 & 0x7FFFFFF)) = &v8[-a1[3]];
        goto LABEL_11;
      }
      if ( v12 <= 0x68000000 )
        goto LABEL_11;
      if ( v12 == 1879048192 )
      {
        *(_QWORD *)(a1[2] + 16 * (v11 & 0x7FFFFFF) + 8) = &v8[-a1[3]];
        goto LABEL_11;
      }
      if ( v13 >= a5 )
        return v8;
LABEL_12:
      v7 = v13;
    }
    v45 = (v11 & 0x7FFFFFF) + v7;
    v13 = v45 + 1;
    v9 = *a1;
    for ( m = a3; ; m = sub_CBAF40(v9, v8, (unsigned __int64)v48, v13, a5) )
    {
      v48 = sub_CBB820((__int64)a1, v8, m, v7, v13);
      if ( a3 == sub_CBB820((__int64)a1, v48, a3, v13, a5) )
        break;
    }
    if ( !sub_CBB820((__int64)a1, v8, v48, v7 + 1, v45) )
    {
      v8 = v48;
      goto LABEL_11;
    }
    v16 = v45;
    v17 = v7 + 1;
    v18 = v48;
LABEL_21:
    sub_CBBCD0(a1, v8, v18, v17, v16);
    v8 = v48;
    goto LABEL_11;
  }
  return v5;
}
