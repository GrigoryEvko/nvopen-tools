// Function: sub_2B812F0
// Address: 0x2b812f0
//
void __fastcall sub_2B812F0(__int64 a1, char **a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // r13
  char **v10; // r12
  __int64 v11; // rbx
  __int64 v12; // r11
  __int64 v13; // r13
  __int64 v14; // r10
  char **v15; // r15
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // r14
  char **v19; // rdx
  __int64 v20; // r12
  __int64 v21; // rax
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r15
  __int64 v26; // r12
  char **v27; // rbx
  char **v28; // rsi
  __int64 v29; // rdi
  __int64 v30; // r15
  __int64 v31; // rbx
  char **v32; // r12
  __int64 v33; // r15
  char **v34; // rsi
  __int64 v35; // rdi
  char **v36; // rsi
  __int64 v37; // rdi
  __int64 v38; // r15
  char **v39; // rsi
  __int64 v40; // rdi
  __int64 v41; // rcx
  __int64 v42; // rax
  char **v43; // rsi
  __int64 v44; // rax
  __int64 v45; // r15
  char **v46; // r12
  __int64 v47; // r15
  __int64 v48; // r12
  char **v49; // rsi
  __int64 v50; // rdi
  char **v51; // r15
  __int64 v52; // rbx
  __int64 v53; // rbx
  char **i; // r15
  __int64 v55; // [rsp-10h] [rbp-70h]
  __int64 v56; // [rsp+8h] [rbp-58h]
  int v57; // [rsp+10h] [rbp-50h]
  __int64 v58; // [rsp+18h] [rbp-48h]
  __int64 v59; // [rsp+18h] [rbp-48h]
  __int64 v60; // [rsp+18h] [rbp-48h]
  __int64 v61; // [rsp+20h] [rbp-40h]
  char **v62; // [rsp+20h] [rbp-40h]
  char **v63; // [rsp+20h] [rbp-40h]
  __int64 v64; // [rsp+28h] [rbp-38h]
  __int64 v65; // [rsp+28h] [rbp-38h]
  __int64 v66; // [rsp+28h] [rbp-38h]
  __int64 v67; // [rsp+28h] [rbp-38h]

  v7 = a5;
  v8 = a3;
  v9 = a1;
  v10 = a2;
  v11 = a6;
  if ( a7 <= a5 )
    v7 = a7;
  if ( v7 >= a4 )
  {
LABEL_22:
    v66 = (__int64)v10 - v9;
    if ( (__int64)v10 - v9 > 0 )
    {
      v62 = v10;
      v25 = ((__int64)v10 - v9) >> 6;
      v26 = v11;
      v59 = v11;
      v27 = (char **)v9;
      do
      {
        v28 = v27;
        v29 = v26;
        v27 += 8;
        v26 += 64;
        sub_2B0F6D0(v29, v28, a3, a4, a5, a6);
        --v25;
      }
      while ( v25 );
      v30 = 64;
      v31 = v59;
      v32 = v62;
      if ( v66 > 0 )
        v30 = v66;
      v33 = v59 + v30;
      if ( (char **)v8 == v62 )
      {
LABEL_53:
        if ( v31 != v33 )
        {
          v47 = v33 - v31;
          v48 = v47 >> 6;
          if ( v47 > 0 )
          {
            do
            {
              v49 = (char **)v31;
              v50 = v9;
              v31 += 64;
              v9 += 64;
              sub_2B0F6D0(v50, v49, a3, a4, a5, a6);
              --v48;
            }
            while ( v48 );
          }
        }
      }
      else if ( v59 != v33 )
      {
        while ( 1 )
        {
          if ( *((_DWORD *)v32 + 2) > *(_DWORD *)(v31 + 8) )
          {
            v34 = v32;
            v35 = v9;
            v32 += 8;
            v9 += 64;
            sub_2B0F6D0(v35, v34, a3, a4, a5, a6);
            if ( v31 == v33 )
              return;
          }
          else
          {
            v36 = (char **)v31;
            v37 = v9;
            v31 += 64;
            v9 += 64;
            sub_2B0F6D0(v37, v36, a3, a4, a5, a6);
            if ( v31 == v33 )
              return;
          }
          if ( (char **)v8 == v32 )
            goto LABEL_53;
        }
      }
    }
  }
  else
  {
    if ( a7 < a5 )
    {
      v12 = a4;
      v13 = a5;
      v56 = a3;
      v14 = a1;
      if ( a5 >= a4 )
        goto LABEL_16;
LABEL_6:
      v15 = a2;
      v16 = v12 / 2;
      v17 = (v56 - (__int64)a2) >> 6;
      v18 = v14 + ((v12 / 2) << 6);
      while ( v17 > 0 )
      {
        while ( 1 )
        {
          v19 = &v15[8 * (v17 >> 1)];
          if ( *((_DWORD *)v19 + 2) <= *(_DWORD *)(v18 + 8) )
            break;
          v15 = v19 + 8;
          v17 = v17 - (v17 >> 1) - 1;
          if ( v17 <= 0 )
            goto LABEL_10;
        }
        v17 >>= 1;
      }
LABEL_10:
      v20 = ((char *)v15 - (char *)a2) >> 6;
      while ( 1 )
      {
        v58 = v12 - v16;
        v57 = v14;
        v13 -= v20;
        v64 = a6;
        v21 = sub_2B80AE0(v18, a2, v15, v12 - v16, v20, a6, a7);
        v22 = v64;
        v65 = v21;
        v61 = v22;
        sub_2B812F0(v57, v18, v21, v16, v20, v22, a7);
        v12 = v58;
        a3 = v13;
        if ( a7 <= v13 )
          a3 = a7;
        a4 = v55;
        a6 = v61;
        if ( a3 >= v58 )
        {
          v8 = v56;
          v11 = v61;
          v10 = v15;
          v9 = v65;
          goto LABEL_22;
        }
        if ( a7 >= v13 )
          break;
        a2 = v15;
        v14 = v65;
        if ( v13 < v58 )
          goto LABEL_6;
LABEL_16:
        v18 = v14;
        v23 = ((__int64)a2 - v14) >> 6;
        v20 = v13 / 2;
        v15 = &a2[8 * (v13 / 2)];
        while ( v23 > 0 )
        {
          while ( 1 )
          {
            v24 = v18 + (v23 >> 1 << 6);
            if ( *((_DWORD *)v15 + 2) > *(_DWORD *)(v24 + 8) )
              break;
            v18 = v24 + 64;
            v23 = v23 - (v23 >> 1) - 1;
            if ( v23 <= 0 )
              goto LABEL_20;
          }
          v23 >>= 1;
        }
LABEL_20:
        v16 = (v18 - v14) >> 6;
      }
      v8 = v56;
      v11 = v61;
      v10 = v15;
      v9 = v65;
    }
    v67 = v8 - (_QWORD)v10;
    if ( v8 - (__int64)v10 > 0 )
    {
      v63 = v10;
      v38 = (v8 - (__int64)v10) >> 6;
      v60 = v11;
      do
      {
        v39 = v10;
        v40 = v11;
        v10 += 8;
        v11 += 64;
        sub_2B0F6D0(v40, v39, a3, a4, a5, a6);
        --v38;
      }
      while ( v38 );
      v41 = v67;
      v42 = v67 - 64;
      if ( v67 <= 0 )
        v42 = 0;
      v43 = (char **)(v60 + v42);
      v44 = 64;
      if ( v67 > 0 )
        v44 = v67;
      v45 = v60 + v44;
      if ( v63 == (char **)v9 )
      {
        v53 = v44 >> 6;
        for ( i = (char **)(v45 - 64); ; v43 = i )
        {
          v8 -= 64;
          i -= 8;
          sub_2B0F6D0(v8, v43, a3, v41, a5, a6);
          if ( !--v53 )
            break;
        }
      }
      else if ( v60 != v45 )
      {
        v46 = v63 - 8;
        while ( 1 )
        {
          v45 -= 64;
          v8 -= 64;
          if ( *(_DWORD *)(v45 + 8) > *((_DWORD *)v46 + 2) )
            break;
LABEL_49:
          sub_2B0F6D0(v8, (char **)v45, a3, v41, a5, a6);
          if ( v60 == v45 )
            return;
        }
        while ( 1 )
        {
          sub_2B0F6D0(v8, v46, a3, v41, a5, a6);
          if ( v46 == (char **)v9 )
            break;
          v46 -= 8;
          v8 -= 64;
          if ( *(_DWORD *)(v45 + 8) <= *((_DWORD *)v46 + 2) )
            goto LABEL_49;
        }
        v51 = (char **)(v45 + 64);
        v52 = ((__int64)v51 - v60) >> 6;
        if ( (__int64)v51 - v60 > 0 )
        {
          do
          {
            v51 -= 8;
            v8 -= 64;
            sub_2B0F6D0(v8, v51, a3, v41, a5, a6);
            --v52;
          }
          while ( v52 );
        }
      }
    }
  }
}
