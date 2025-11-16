// Function: sub_27AF640
// Address: 0x27af640
//
__int64 __fastcall sub_27AF640(__int64 *a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 *v5; // r12
  __int64 v6; // r11
  __int64 v7; // r13
  int v8; // edx
  __int64 v9; // rsi
  int v10; // edx
  unsigned int v11; // ecx
  __int64 *v12; // rax
  __int64 v13; // r8
  unsigned int v14; // r8d
  unsigned int v15; // ecx
  __int64 *v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rbx
  __int64 v19; // rdx
  __int64 v20; // r11
  __int64 v21; // r11
  __int64 v22; // r15
  __int64 v23; // rcx
  __int64 *v24; // r14
  __int64 v25; // rbx
  __int64 *v26; // r9
  int v27; // eax
  __int64 *v28; // rsi
  int v29; // r10d
  unsigned int v30; // edi
  __int64 *v31; // rdx
  __int64 v32; // r8
  unsigned int v33; // r8d
  __int64 v34; // r13
  _QWORD *v35; // rdi
  __int64 v36; // r12
  _QWORD *v37; // rdx
  __int64 *v38; // rax
  __int64 v39; // r9
  unsigned int v40; // edx
  int i; // eax
  unsigned int v42; // r9d
  unsigned int v43; // edx
  __int64 *v44; // rax
  __int64 v45; // r8
  int v46; // eax
  int v47; // edx
  int v48; // eax
  __int64 v49; // rdx
  int v50; // edx
  int v51; // r8d
  int v52; // eax
  bool v53; // al
  __int64 v54; // rdx
  __int64 v55; // rbx
  __int64 v56; // r12
  __int64 v57; // r8
  __int64 *v58; // rbx
  __int64 v59; // rcx
  __int64 v60; // rdx
  __int64 v61; // rdx
  int v62; // r12d
  int v63; // eax
  int v64; // r9d
  int v65; // edi
  __int64 *v66; // [rsp+10h] [rbp-70h]
  int v67; // [rsp+18h] [rbp-68h]
  unsigned int v68; // [rsp+1Ch] [rbp-64h]
  __int64 v69; // [rsp+20h] [rbp-60h]
  int v70; // [rsp+28h] [rbp-58h]
  __int64 v71; // [rsp+28h] [rbp-58h]
  __int64 *v72; // [rsp+30h] [rbp-50h]
  __int64 *v75; // [rsp+48h] [rbp-38h]

  result = a2 - (char *)a1;
  v72 = (__int64 *)a2;
  v69 = a3;
  if ( a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v75 = (__int64 *)a2;
    goto LABEL_49;
  }
  v66 = a1 + 2;
  while ( 2 )
  {
    --v69;
    v5 = &a1[result >> 4];
    v6 = a1[1];
    v7 = *v5;
    v8 = *(_DWORD *)(a4 + 592);
    v9 = *(_QWORD *)(a4 + 576);
    if ( !v8 )
      goto LABEL_46;
    v10 = v8 - 1;
    v11 = v10 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v12 = (__int64 *)(v9 + 16LL * v11);
    v13 = *v12;
    if ( *v12 == v6 )
    {
LABEL_6:
      v14 = *((_DWORD *)v12 + 2);
    }
    else
    {
      v63 = 1;
      while ( v13 != -4096 )
      {
        v65 = v63 + 1;
        v11 = v10 & (v63 + v11);
        v12 = (__int64 *)(v9 + 16LL * v11);
        v13 = *v12;
        if ( v6 == *v12 )
          goto LABEL_6;
        v63 = v65;
      }
      v14 = 0;
    }
    v15 = v10 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v16 = (__int64 *)(v9 + 16LL * v15);
    v17 = *v16;
    if ( v7 != *v16 )
    {
      v52 = 1;
      while ( v17 != -4096 )
      {
        v64 = v52 + 1;
        v15 = v10 & (v52 + v15);
        v16 = (__int64 *)(v9 + 16LL * v15);
        v17 = *v16;
        if ( v7 == *v16 )
          goto LABEL_8;
        v52 = v64;
      }
      goto LABEL_46;
    }
LABEL_8:
    if ( v14 >= *((_DWORD *)v16 + 2) )
    {
LABEL_46:
      v53 = sub_27ACFF0(a4, a1[1], *(v72 - 1));
      v22 = *a1;
      if ( v53 )
      {
        *a1 = v21;
        a1[1] = v22;
        v23 = *(v72 - 1);
      }
      else if ( sub_27ACFF0(a4, v7, v54) )
      {
        *a1 = v60;
        v23 = v22;
        *(v72 - 1) = v22;
        v21 = *a1;
        v22 = a1[1];
      }
      else
      {
        *a1 = v7;
        *v5 = v22;
        v21 = *a1;
        v22 = a1[1];
        v23 = *(v72 - 1);
      }
      goto LABEL_11;
    }
    v18 = *a1;
    if ( sub_27ACFF0(a4, *v5, *(v72 - 1)) )
    {
      *a1 = v7;
      *v5 = v18;
      v21 = *a1;
      v22 = a1[1];
      v23 = *(v72 - 1);
    }
    else if ( sub_27ACFF0(a4, v20, v19) )
    {
      *a1 = v61;
      v23 = v18;
      *(v72 - 1) = v18;
      v21 = *a1;
      v22 = a1[1];
    }
    else
    {
      *a1 = v21;
      v22 = v18;
      a1[1] = v18;
      v23 = *(v72 - 1);
    }
LABEL_11:
    v24 = v66;
    v25 = *(_QWORD *)(a4 + 576);
    v26 = v72 - 1;
    v27 = *(_DWORD *)(a4 + 592);
    while ( 1 )
    {
      v75 = v24 - 1;
      v28 = v26;
      if ( !v27 )
        break;
      v29 = v27 - 1;
      v30 = (v27 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v31 = (__int64 *)(v25 + 16LL * v30);
      v32 = *v31;
      if ( v22 == *v31 )
      {
LABEL_15:
        v33 = *((_DWORD *)v31 + 2);
      }
      else
      {
        v50 = 1;
        while ( v32 != -4096 )
        {
          v62 = v50 + 1;
          v30 = v29 & (v50 + v30);
          v31 = (__int64 *)(v25 + 16LL * v30);
          v32 = *v31;
          if ( v22 == *v31 )
            goto LABEL_15;
          v50 = v62;
        }
        v33 = 0;
      }
      v34 = v29 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v35 = (_QWORD *)(v25 + 16 * v34);
      v36 = *v35;
      v37 = v35;
      if ( v21 != *v35 )
      {
        v71 = *v35;
        v47 = 1;
        v68 = v29 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v67 = v27;
        while ( v71 != -4096 )
        {
          v48 = v47 + 1;
          v49 = v29 & (v68 + v47);
          v68 = v49;
          v37 = (_QWORD *)(v25 + 16 * v49);
          v71 = *v37;
          if ( v21 == *v37 )
          {
            v27 = v67;
            goto LABEL_17;
          }
          v47 = v48;
        }
        while ( 1 )
        {
          v38 = (__int64 *)(v25 + 16 * v34);
          if ( v21 != v36 )
            goto LABEL_19;
LABEL_26:
          v42 = *((_DWORD *)v38 + 2);
LABEL_27:
          v43 = v29 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v44 = (__int64 *)(v25 + 16LL * v43);
          v45 = *v44;
          if ( *v44 != v23 )
            break;
LABEL_23:
          if ( v42 >= *((_DWORD *)v44 + 2) )
            goto LABEL_30;
          v23 = *--v28;
        }
        v46 = 1;
        while ( v45 != -4096 )
        {
          v43 = v29 & (v46 + v43);
          v70 = v46 + 1;
          v44 = (__int64 *)(v25 + 16LL * v43);
          v45 = *v44;
          if ( v23 == *v44 )
            goto LABEL_23;
          v46 = v70;
        }
        break;
      }
LABEL_17:
      if ( v33 >= *((_DWORD *)v37 + 2) )
      {
        v38 = (__int64 *)(v25 + 16 * v34);
        if ( v21 == v36 )
          goto LABEL_26;
LABEL_19:
        v39 = *v35;
        v40 = v29 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        for ( i = 1; ; i = v51 )
        {
          if ( v39 == -4096 )
          {
            v42 = 0;
            goto LABEL_27;
          }
          v51 = i + 1;
          v40 = v29 & (i + v40);
          v38 = (__int64 *)(v25 + 16LL * v40);
          v39 = *v38;
          if ( v21 == *v38 )
            break;
        }
        goto LABEL_26;
      }
LABEL_12:
      v22 = *v24++;
    }
LABEL_30:
    if ( v75 < v28 )
    {
      *(v24 - 1) = v23;
      v26 = v28 - 1;
      *v28 = v22;
      v23 = *(v28 - 1);
      v21 = *a1;
      v25 = *(_QWORD *)(a4 + 576);
      v27 = *(_DWORD *)(a4 + 592);
      goto LABEL_12;
    }
    sub_27AF640(v75, v72, v69, a4);
    result = (char *)v75 - (char *)a1;
    if ( (char *)v75 - (char *)a1 > 128 )
    {
      if ( v69 )
      {
        v72 = v24 - 1;
        continue;
      }
LABEL_49:
      v55 = result >> 3;
      v56 = ((result >> 3) - 2) >> 1;
      sub_27AD9C0((__int64)a1, v56, result >> 3, a1[v56], a4);
      do
      {
        --v56;
        sub_27AD9C0((__int64)a1, v56, v55, a1[v56], v57);
      }
      while ( v56 );
      v58 = v75;
      do
      {
        v59 = *--v58;
        *v58 = *a1;
        result = sub_27AD9C0((__int64)a1, 0, v58 - a1, v59, v57);
      }
      while ( (char *)v58 - (char *)a1 > 8 );
    }
    return result;
  }
}
