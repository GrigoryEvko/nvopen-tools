// Function: sub_2959530
// Address: 0x2959530
//
__int64 __fastcall sub_2959530(__int64 *a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 *v6; // rbx
  __int64 v7; // r14
  __int64 v8; // r13
  bool v9; // al
  __int64 v10; // r12
  char v11; // r11
  __int64 *v12; // r13
  __int64 *v13; // rsi
  char v14; // di
  __int64 v15; // rbx
  int v16; // r10d
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // rcx
  _QWORD *v20; // rdx
  unsigned int j; // eax
  __int64 *v22; // r9
  __int64 *v23; // rdx
  unsigned int v24; // ecx
  _QWORD *m; // rdx
  __int64 v26; // rax
  int v27; // r10d
  unsigned int v28; // edx
  __int64 *v29; // rax
  __int64 v30; // r11
  unsigned int v31; // edx
  _QWORD *ii; // rax
  __int64 v33; // rcx
  __int64 *v34; // rax
  unsigned int v35; // r8d
  _QWORD *n; // rax
  int v37; // eax
  unsigned int v38; // edx
  __int64 v39; // r11
  int v40; // eax
  int v41; // r8d
  unsigned int v42; // ecx
  __int64 v43; // r8
  int k; // edx
  int i; // eax
  int v46; // r8d
  __int64 v47; // rax
  __int64 v48; // rbx
  __int64 v49; // r12
  __int64 *v50; // rbx
  __int64 v51; // rcx
  __int64 *v52; // rax
  bool v53; // zf
  __int64 *v54; // [rsp+0h] [rbp-70h]
  __int64 v55; // [rsp+8h] [rbp-68h]
  __int64 *v56; // [rsp+10h] [rbp-60h]
  __int64 v57; // [rsp+18h] [rbp-58h]
  int v59; // [rsp+28h] [rbp-48h]
  int v60; // [rsp+28h] [rbp-48h]
  unsigned int v61; // [rsp+2Ch] [rbp-44h]
  char *v62; // [rsp+30h] [rbp-40h]
  __int64 v63; // [rsp+38h] [rbp-38h]

  result = a2 - (char *)a1;
  v56 = (__int64 *)a2;
  v55 = a3;
  if ( a2 - (char *)a1 > 128 )
  {
    if ( a3 )
    {
      v54 = a1 + 2;
      v57 = a4 + 16;
      while ( 1 )
      {
        --v55;
        v6 = &a1[result >> 4];
        v7 = a1[1];
        v8 = *v6;
        v9 = sub_2959010(a4, v7, *v6);
        v10 = *(v56 - 1);
        v63 = *a1;
        if ( !v9 )
          break;
        if ( !sub_2959010(a4, v8, v10) )
        {
          v53 = !sub_2959010(a4, v7, v10);
          v52 = a1;
          if ( !v53 )
            goto LABEL_63;
LABEL_61:
          *v52 = v7;
          v52[1] = v63;
          goto LABEL_7;
        }
        *a1 = v8;
        *v6 = v63;
        v7 = *a1;
        v63 = a1[1];
LABEL_7:
        v11 = *(_BYTE *)(a4 + 8);
        v12 = v54;
        v13 = v56;
        v62 = (char *)(v54 - 1);
        v14 = v11 & 1;
        if ( (v11 & 1) != 0 )
        {
LABEL_8:
          v15 = v57;
          v16 = 15;
          goto LABEL_9;
        }
        while ( 1 )
        {
          v27 = *(_DWORD *)(a4 + 24);
          v15 = *(_QWORD *)(a4 + 16);
          if ( !v27 )
LABEL_35:
            BUG();
          v16 = v27 - 1;
LABEL_9:
          v17 = v16 & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
          v18 = (__int64 *)(v15 + 16LL * v17);
          v19 = *v18;
          if ( *v18 != v63 )
          {
            for ( i = 1; ; i = v46 )
            {
              if ( v19 == -4096 )
                goto LABEL_66;
              v46 = i + 1;
              v47 = v16 & (v17 + i);
              v17 = v47;
              v18 = (__int64 *)(v15 + 16 * v47);
              v19 = *v18;
              if ( *v18 == v63 )
                break;
            }
          }
          v20 = *(_QWORD **)v18[1];
          for ( j = 1; v20; ++j )
            v20 = (_QWORD *)*v20;
          if ( !v14 && !*(_DWORD *)(a4 + 24) )
            goto LABEL_66;
          v61 = v16 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
          v22 = (__int64 *)(v15 + 16LL * v61);
          v23 = v22;
          if ( *v22 != v7 )
          {
            v42 = v16 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
            v43 = *v22;
            for ( k = 1; ; k = v60 )
            {
              if ( v43 == -4096 )
                goto LABEL_66;
              v42 = v16 & (k + v42);
              v60 = k + 1;
              v23 = (__int64 *)(v15 + 16LL * v42);
              v43 = *v23;
              if ( *v23 == v7 )
                break;
            }
          }
          v24 = 1;
          for ( m = *(_QWORD **)v23[1]; m; ++v24 )
            m = (_QWORD *)*m;
          if ( j >= v24 )
            break;
LABEL_18:
          v26 = *v12++;
          v63 = v26;
          v62 = (char *)(v12 - 1);
          v14 = v11 & 1;
          if ( (v11 & 1) != 0 )
            goto LABEL_8;
        }
        do
        {
          v33 = *--v13;
          if ( !v14 && !*(_DWORD *)(a4 + 24) )
            goto LABEL_35;
          v34 = (__int64 *)(v15 + 16LL * v61);
          if ( *v22 != v7 )
          {
            v38 = v16 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
            v39 = *v22;
            v40 = 1;
            while ( v39 != -4096 )
            {
              v41 = v40 + 1;
              v38 = v16 & (v40 + v38);
              v34 = (__int64 *)(v15 + 16LL * v38);
              v39 = *v34;
              if ( *v34 == v7 )
                goto LABEL_30;
              v40 = v41;
            }
LABEL_66:
            BUG();
          }
LABEL_30:
          v35 = 1;
          for ( n = *(_QWORD **)v34[1]; n; ++v35 )
            n = (_QWORD *)*n;
          if ( !v14 && !*(_DWORD *)(a4 + 24) )
            goto LABEL_66;
          v28 = v16 & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
          v29 = (__int64 *)(v15 + 16LL * v28);
          v30 = *v29;
          if ( v33 != *v29 )
          {
            v37 = 1;
            while ( v30 != -4096 )
            {
              v28 = v16 & (v37 + v28);
              v59 = v37 + 1;
              v29 = (__int64 *)(v15 + 16LL * v28);
              v30 = *v29;
              if ( v33 == *v29 )
                goto LABEL_24;
              v37 = v59;
            }
            goto LABEL_66;
          }
LABEL_24:
          v31 = 1;
          for ( ii = *(_QWORD **)v29[1]; ii; ++v31 )
            ii = (_QWORD *)*ii;
        }
        while ( v35 < v31 );
        if ( v13 > (__int64 *)v62 )
        {
          *(v12 - 1) = v33;
          *v13 = v63;
          v11 = *(_BYTE *)(a4 + 8);
          v7 = *a1;
          goto LABEL_18;
        }
        sub_2959530(v62, v56, v55, a4);
        result = v62 - (char *)a1;
        if ( v62 - (char *)a1 <= 128 )
          return result;
        v56 = (__int64 *)v62;
        if ( !v55 )
          goto LABEL_54;
      }
      if ( !sub_2959010(a4, v7, v10) )
      {
        v53 = !sub_2959010(a4, v8, v10);
        v52 = a1;
        if ( v53 )
        {
          *a1 = v8;
          *v6 = v63;
          v7 = *a1;
          v63 = a1[1];
          goto LABEL_7;
        }
LABEL_63:
        *v52 = v10;
        *(v56 - 1) = v63;
        v7 = *v52;
        v63 = v52[1];
        goto LABEL_7;
      }
      v52 = a1;
      goto LABEL_61;
    }
LABEL_54:
    v48 = result >> 3;
    v49 = ((result >> 3) - 2) >> 1;
    sub_29593A0((__int64)a1, v49, result >> 3, a1[v49], a4);
    do
    {
      --v49;
      sub_29593A0((__int64)a1, v49, v48, a1[v49], a4);
    }
    while ( v49 );
    v50 = v56;
    do
    {
      v51 = *--v50;
      *v50 = *a1;
      result = sub_29593A0((__int64)a1, 0, v50 - a1, v51, a4);
    }
    while ( (char *)v50 - (char *)a1 > 8 );
  }
  return result;
}
