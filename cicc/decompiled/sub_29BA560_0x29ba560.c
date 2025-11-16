// Function: sub_29BA560
// Address: 0x29ba560
//
signed __int64 __fastcall sub_29BA560(char *a1, char *a2, __int64 a3)
{
  signed __int64 result; // rax
  __int64 v4; // r14
  char *v5; // r12
  char *v7; // r10
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdi
  char v11; // r11
  __int64 v12; // r9
  __int64 v13; // rcx
  char v14; // r8
  __int64 v15; // r13
  char v16; // r12
  __int64 v17; // r13
  double v18; // xmm1_8
  __int64 v19; // rdi
  double v20; // xmm2_8
  double v21; // xmm0_8
  double v22; // xmm2_8
  double v23; // xmm1_8
  __int64 v24; // rax
  __int64 v25; // r11
  __int64 v26; // rdi
  __int64 v27; // r13
  double v28; // xmm1_8
  double v29; // xmm0_8
  double v30; // xmm3_8
  __int64 *v31; // r10
  __int64 v32; // r9
  char *v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // rcx
  double v36; // xmm0_8
  __int64 v37; // rcx
  double v38; // xmm1_8
  double v39; // xmm2_8
  double v40; // xmm0_8
  __int64 v41; // rax
  double v42; // xmm0_8
  __int64 v43; // rax
  double v44; // xmm2_8
  double v45; // xmm0_8
  double v46; // xmm1_8
  __int64 v47; // r13
  __int64 v48; // r13
  __int64 v49; // rcx
  double v50; // xmm0_8
  __int64 v51; // rcx
  double v52; // xmm1_8
  double v53; // xmm2_8
  double v54; // xmm0_8
  __int64 v55; // r13
  __int64 v56; // r13
  __int64 v57; // r14
  __int64 i; // r13
  __int64 *v59; // r12
  __int64 v60; // rcx
  __int64 v61; // r13
  __int64 v62; // r8
  double v63; // xmm0_8
  __int64 v64; // r8
  double v65; // xmm1_8
  double v66; // xmm2_8
  double v67; // xmm0_8
  __int64 v68; // r15
  __int64 v69; // r9
  __int64 v70; // r11
  __int64 v71; // rdx
  double v72; // xmm0_8
  __int64 v73; // rdx
  double v74; // xmm2_8
  double v75; // xmm0_8
  double v76; // xmm1_8
  __int64 v77; // r9
  __int64 v78; // rdi
  __int64 v79; // rdi
  __int64 v80; // rdi
  __int64 v81; // rdi
  __int64 *v82; // [rsp-40h] [rbp-40h]

  result = a2 - a1;
  if ( a2 - a1 <= 128 )
    return result;
  v4 = a3;
  v5 = a2;
  if ( !a3 )
    goto LABEL_69;
  v82 = (__int64 *)(a1 + 16);
  while ( 2 )
  {
    --v4;
    v7 = &a1[8 * ((__int64)(((a2 - a1) >> 3) + ((unsigned __int64)(a2 - a1) >> 63)) >> 1)];
    v8 = *((_QWORD *)a1 + 1);
    v9 = *(_QWORD *)v7;
    v10 = ***(_QWORD ***)(v8 + 32);
    v11 = v10 == 0;
    v12 = ***(_QWORD ***)(*(_QWORD *)v7 + 32LL);
    v13 = *((_QWORD *)a2 - 1);
    v14 = v12 == 0;
    v15 = ***(_QWORD ***)(v13 + 32);
    v16 = v15 == 0;
    if ( (v10 == 0) == (v12 == 0) )
    {
      v27 = *(_QWORD *)(v9 + 24);
      if ( v27 < 0 )
      {
        v68 = *(_QWORD *)(v9 + 24) & 1LL | (*(_QWORD *)(v9 + 24) >> 1);
        v28 = (double)(int)v68 + (double)(int)v68;
      }
      else
      {
        v28 = (double)(int)v27;
      }
      v17 = *(_QWORD *)(v8 + 24);
      v29 = *(double *)(v9 + 16) / v28;
      if ( v17 < 0 )
        v30 = (double)(int)(*(_DWORD *)(v8 + 24) & 1 | ((unsigned __int64)v17 >> 1))
            + (double)(int)(*(_DWORD *)(v8 + 24) & 1 | ((unsigned __int64)v17 >> 1));
      else
        v30 = (double)(int)v17;
      v18 = *(double *)(v8 + 16);
      if ( v18 / v30 <= v29 && (v29 > v18 / v30 || *(_QWORD *)v9 <= *(_QWORD *)v8) )
      {
        if ( v11 != v16 )
        {
          if ( v10 )
          {
LABEL_15:
            if ( v12 )
            {
LABEL_16:
              v24 = *(_QWORD *)a1;
              *(_QWORD *)a1 = v9;
              *(_QWORD *)v7 = v24;
              v8 = *(_QWORD *)a1;
              v25 = *((_QWORD *)a1 + 1);
              v26 = *((_QWORD *)a2 - 1);
              goto LABEL_29;
            }
            goto LABEL_53;
          }
          goto LABEL_28;
        }
        goto LABEL_8;
      }
LABEL_24:
      if ( v14 == v16 )
      {
        v62 = *(_QWORD *)(v13 + 24);
        if ( v62 < 0 )
        {
          v69 = *(_QWORD *)(v13 + 24) & 1LL | (*(_QWORD *)(v13 + 24) >> 1);
          v63 = (double)(int)v69 + (double)(int)v69;
        }
        else
        {
          v63 = (double)(int)v62;
        }
        v64 = *(_QWORD *)(v9 + 24);
        v65 = *(double *)(v13 + 16) / v63;
        if ( v64 < 0 )
        {
          v77 = *(_QWORD *)(v9 + 24) & 1LL | (*(_QWORD *)(v9 + 24) >> 1);
          v66 = (double)(int)v77 + (double)(int)v77;
        }
        else
        {
          v66 = (double)(int)v64;
        }
        v67 = *(double *)(v9 + 16) / v66;
        if ( v67 > v65 || v65 <= v67 && *(_QWORD *)v13 > *(_QWORD *)v9 )
          goto LABEL_16;
      }
      else if ( !v12 )
      {
        goto LABEL_16;
      }
      if ( v11 == v16 )
      {
        v71 = *(_QWORD *)(v13 + 24);
        if ( v71 < 0 )
        {
          v78 = *(_QWORD *)(v13 + 24) & 1LL | (*(_QWORD *)(v13 + 24) >> 1);
          v72 = (double)(int)v78 + (double)(int)v78;
        }
        else
        {
          v72 = (double)(int)v71;
        }
        v73 = *(_QWORD *)(v8 + 24);
        v74 = *(double *)(v13 + 16) / v72;
        if ( v73 < 0 )
        {
          v81 = *(_QWORD *)(v8 + 24) & 1LL | (*(_QWORD *)(v8 + 24) >> 1);
          v75 = (double)(int)v81 + (double)(int)v81;
        }
        else
        {
          v75 = (double)(int)v73;
        }
        v76 = *(double *)(v8 + 16) / v75;
        if ( v76 > v74 || v74 <= v76 && *(_QWORD *)v13 > *(_QWORD *)v8 )
          goto LABEL_53;
      }
      else if ( !v10 )
      {
        goto LABEL_53;
      }
      goto LABEL_28;
    }
    if ( !v10 )
      goto LABEL_24;
    if ( v15 )
    {
      v17 = *(_QWORD *)(v8 + 24);
      v18 = *(double *)(v8 + 16);
LABEL_8:
      v19 = *(_QWORD *)(v13 + 24);
      if ( v19 < 0 )
      {
        v70 = *(_QWORD *)(v13 + 24) & 1LL | (*(_QWORD *)(v13 + 24) >> 1);
        v20 = (double)(int)v70 + (double)(int)v70;
      }
      else
      {
        v20 = (double)(int)v19;
      }
      v21 = *(double *)(v13 + 16) / v20;
      if ( v17 < 0 )
        v22 = (double)(int)(v17 & 1 | ((unsigned __int64)v17 >> 1))
            + (double)(int)(v17 & 1 | ((unsigned __int64)v17 >> 1));
      else
        v22 = (double)(int)v17;
      v23 = v18 / v22;
      if ( v23 <= v21 && (v21 > v23 || *(_QWORD *)v13 <= *(_QWORD *)v8) )
        goto LABEL_14;
LABEL_28:
      v25 = *(_QWORD *)a1;
      *(_QWORD *)a1 = v8;
      *((_QWORD *)a1 + 1) = v25;
      v26 = *((_QWORD *)a2 - 1);
      goto LABEL_29;
    }
LABEL_14:
    if ( v14 != v16 )
      goto LABEL_15;
    v41 = *(_QWORD *)(v13 + 24);
    if ( v41 < 0 )
    {
      v79 = *(_QWORD *)(v13 + 24) & 1LL | (*(_QWORD *)(v13 + 24) >> 1);
      v42 = (double)(int)v79 + (double)(int)v79;
    }
    else
    {
      v42 = (double)(int)v41;
    }
    v43 = *(_QWORD *)(v9 + 24);
    v44 = *(double *)(v13 + 16) / v42;
    if ( v43 < 0 )
    {
      v80 = *(_QWORD *)(v9 + 24) & 1LL | (*(_QWORD *)(v9 + 24) >> 1);
      v45 = (double)(int)v80 + (double)(int)v80;
    }
    else
    {
      v45 = (double)(int)v43;
    }
    v46 = *(double *)(v9 + 16) / v45;
    if ( v46 <= v44 && (v44 > v46 || *(_QWORD *)v13 <= *(_QWORD *)v9) )
      goto LABEL_16;
LABEL_53:
    v26 = *(_QWORD *)a1;
    *(_QWORD *)a1 = v13;
    *((_QWORD *)a2 - 1) = v26;
    v8 = *(_QWORD *)a1;
    v25 = *((_QWORD *)a1 + 1);
LABEL_29:
    v31 = v82;
    v32 = ***(_QWORD ***)(v8 + 32);
    v33 = a2;
    while ( 1 )
    {
      v5 = (char *)(v31 - 1);
      v34 = ***(_QWORD ***)(v25 + 32);
      if ( (v34 == 0) == (v32 == 0) )
        break;
      if ( v34 )
        goto LABEL_32;
LABEL_45:
      v25 = *v31++;
    }
    v49 = *(_QWORD *)(v8 + 24);
    if ( v49 < 0 )
    {
      v55 = *(_QWORD *)(v8 + 24) & 1LL | (*(_QWORD *)(v8 + 24) >> 1);
      v50 = (double)(int)v55 + (double)(int)v55;
    }
    else
    {
      v50 = (double)(int)v49;
    }
    v51 = *(_QWORD *)(v25 + 24);
    v52 = *(double *)(v8 + 16) / v50;
    if ( v51 < 0 )
    {
      v56 = *(_QWORD *)(v25 + 24) & 1LL | (*(_QWORD *)(v25 + 24) >> 1);
      v53 = (double)(int)v56 + (double)(int)v56;
    }
    else
    {
      v53 = (double)(int)v51;
    }
    v54 = *(double *)(v25 + 16) / v53;
    if ( v54 > v52 || v52 <= v54 && *(_QWORD *)v8 > *(_QWORD *)v25 )
      goto LABEL_45;
LABEL_32:
    v33 -= 8;
    while ( (v32 == 0) != (***(_QWORD ***)(v26 + 32) == 0) )
    {
      if ( v32 )
        goto LABEL_43;
LABEL_34:
      v26 = *((_QWORD *)v33 - 1);
      v33 -= 8;
    }
    v35 = *(_QWORD *)(v26 + 24);
    if ( v35 < 0 )
    {
      v48 = *(_QWORD *)(v26 + 24) & 1LL | (*(_QWORD *)(v26 + 24) >> 1);
      v36 = (double)(int)v48 + (double)(int)v48;
    }
    else
    {
      v36 = (double)(int)v35;
    }
    v37 = *(_QWORD *)(v8 + 24);
    v38 = *(double *)(v26 + 16) / v36;
    if ( v37 < 0 )
    {
      v47 = *(_QWORD *)(v8 + 24) & 1LL | (*(_QWORD *)(v8 + 24) >> 1);
      v39 = (double)(int)v47 + (double)(int)v47;
    }
    else
    {
      v39 = (double)(int)v37;
    }
    v40 = *(double *)(v8 + 16) / v39;
    if ( v40 > v38 || v38 <= v40 && *(_QWORD *)v26 > *(_QWORD *)v8 )
      goto LABEL_34;
LABEL_43:
    if ( v33 > v5 )
    {
      *(v31 - 1) = v26;
      v26 = *((_QWORD *)v33 - 1);
      *(_QWORD *)v33 = v25;
      v8 = *(_QWORD *)a1;
      v32 = ***(_QWORD ***)(*(_QWORD *)a1 + 32LL);
      goto LABEL_45;
    }
    sub_29BA560(v31 - 1, a2, v4);
    result = v5 - a1;
    if ( v5 - a1 > 128 )
    {
      if ( v4 )
      {
        a2 = v5;
        continue;
      }
LABEL_69:
      v57 = result >> 3;
      for ( i = ((result >> 3) - 2) >> 1; ; --i )
      {
        sub_29B9A00((__int64)a1, i, v57, *(_QWORD *)&a1[8 * i]);
        if ( !i )
          break;
      }
      v59 = (__int64 *)(v5 - 8);
      do
      {
        v60 = *v59;
        v61 = (char *)v59-- - a1;
        v59[1] = *(_QWORD *)a1;
        result = (signed __int64)sub_29B9A00((__int64)a1, 0, v61 >> 3, v60);
      }
      while ( v61 > 8 );
    }
    return result;
  }
}
