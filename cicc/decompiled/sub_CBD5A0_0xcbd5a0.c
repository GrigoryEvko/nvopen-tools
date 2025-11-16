// Function: sub_CBD5A0
// Address: 0xcbd5a0
//
char *__fastcall sub_CBD5A0(__int64 *a1, char *a2, char *a3, __int64 a4, __int64 a5)
{
  char *v5; // r13
  char *v7; // r15
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // rcx
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 v15; // rax
  char *v16; // r13
  __int64 v17; // r15
  char *v18; // rdx
  __int64 v19; // rbx
  __int64 v20; // rbx
  char *j; // rax
  char *v22; // r14
  __int64 v23; // rax
  char *v24; // r11
  __int64 *v25; // r14
  char *v26; // r12
  __int64 v27; // rbx
  __int64 k; // r13
  __int64 v29; // rax
  __int64 v30; // r13
  char *v31; // r11
  __int64 v32; // rcx
  __int64 v33; // r8
  char *v34; // rdx
  char *i; // rax
  char *v36; // rdx
  char *v37; // r14
  __int64 v38; // r13
  char *v39; // rax
  char *v40; // r11
  __int64 v41; // [rsp+0h] [rbp-60h]
  __int64 v42; // [rsp+8h] [rbp-58h]
  __int64 v43; // [rsp+10h] [rbp-50h]
  __int64 v44; // [rsp+10h] [rbp-50h]
  __int64 v45; // [rsp+18h] [rbp-48h]
  char *v46; // [rsp+20h] [rbp-40h]
  __int64 v47; // [rsp+20h] [rbp-40h]
  char *v48; // [rsp+20h] [rbp-40h]
  unsigned __int64 v49; // [rsp+20h] [rbp-40h]
  char *v50; // [rsp+20h] [rbp-40h]

  v5 = a2;
  if ( a4 < a5 )
  {
    v7 = a2;
    v8 = a4;
    v9 = a5;
    while ( 1 )
    {
      v10 = *(_QWORD *)(*a1 + 8);
      v11 = *(_QWORD *)(v10 + 8 * v8);
      v12 = (unsigned int)v11 & 0xF8000000;
      if ( v12 == 1476395008 )
      {
        v44 = v8 + (v11 & 0x7FFFFFF);
        v13 = v44 + 1;
        v41 = v8 + 1;
        v15 = v8;
        v18 = a3;
        v16 = v7;
        v17 = v15;
        while ( 1 )
        {
          v46 = sub_CBD060(a1, v16, v18, v17, v13);
          if ( a3 == sub_CBD060(a1, v46, a3, v13, v9) )
            break;
          v18 = sub_CBAF40(*a1, v16, (unsigned __int64)v46, v13, v9);
        }
        if ( sub_CBD060(a1, v16, v46, v41, v44) )
          sub_CBD5A0(a1, v16, v46, v41, v44);
        v7 = v46;
LABEL_11:
        if ( v13 >= v9 )
          return v7;
        goto LABEL_12;
      }
      if ( v12 == 2013265920 )
        break;
      if ( v12 == 1207959552 )
      {
        v43 = v8 + (v11 & 0x7FFFFFF);
        v13 = v43 + 1;
        for ( i = sub_CBD060(a1, v7, a3, v8, v43 + 1); ; i = sub_CBD060(a1, v7, v36, v8, v13) )
        {
          v49 = (unsigned __int64)i;
          if ( a3 == sub_CBD060(a1, i, a3, v13, v9) )
            break;
          v36 = sub_CBAF40(*a1, v7, v49, v13, v9);
        }
        v42 = v9;
        v37 = (char *)v49;
        v50 = v7;
        v38 = v8 + 1;
        while ( 1 )
        {
          v39 = sub_CBD060(a1, v7, v37, v38, v43);
          if ( !v39 )
            break;
          if ( v7 == v39 )
          {
            v40 = v37;
            v9 = v42;
            goto LABEL_43;
          }
          v50 = v7;
          v7 = v39;
        }
        v40 = v37;
        v9 = v42;
        v39 = v7;
        v7 = v50;
LABEL_43:
        v48 = v40;
        v33 = v43;
        v32 = v38;
        v34 = v39;
        goto LABEL_34;
      }
      v13 = v8 + 1;
      if ( v12 <= 0x48000000 )
      {
        if ( v12 == 671088640 || (v11 & 0xD8000000) == 0x10000000 )
          ++v7;
        goto LABEL_11;
      }
      if ( v12 == 1744830464 )
      {
        *(_QWORD *)(a1[2] + 16 * (v11 & 0x7FFFFFF)) = &v7[-a1[3]];
        goto LABEL_11;
      }
      if ( v12 <= 0x68000000 )
        goto LABEL_11;
      if ( v12 == 1879048192 )
      {
        *(_QWORD *)(a1[2] + 16 * (v11 & 0x7FFFFFF) + 8) = &v7[-a1[3]];
        goto LABEL_11;
      }
      if ( v13 >= v9 )
        return v7;
LABEL_12:
      v8 = v13;
    }
    v19 = v8;
    do
    {
      v19 += v11 & 0x7FFFFFF;
      v11 = *(_QWORD *)(v10 + 8 * v19);
    }
    while ( (v11 & 0xF8000000) != 0x90000000LL );
    v20 = v19 + 1;
    v45 = v9;
    for ( j = sub_CBD060(a1, v7, a3, v8, v20); ; j = sub_CBD060(a1, v7, v22 - 1, v8, v20) )
    {
      v22 = j;
      if ( a3 == sub_CBD060(a1, j, a3, v20, v45) )
        break;
    }
    v23 = *a1;
    v24 = v22;
    v25 = a1;
    v47 = v20;
    v26 = v24;
    v27 = v8 + 1;
    for ( k = v8 + (*(_QWORD *)(*(_QWORD *)(v23 + 8) + 8 * v8) & 0x7FFFFFFLL) - 1;
          v26 != sub_CBD060(v25, v7, v26, v27, k);
          k = v30 - ((*(_QWORD *)(v29 + 8 * v30) & 0xF8000000LL) == 2281701376LL) )
    {
      v27 = k + 2;
      v29 = *(_QWORD *)(*v25 + 8);
      v30 = k + 1 + (*(_QWORD *)(v29 + 8 * (k + 1)) & 0x7FFFFFFLL);
    }
    v31 = v26;
    v32 = v27;
    v13 = v47;
    a1 = v25;
    v48 = v31;
    v9 = v45;
    v33 = k;
    v34 = v31;
LABEL_34:
    sub_CBD5A0(a1, v7, v34, v32, v33);
    v7 = v48;
    goto LABEL_11;
  }
  return v5;
}
