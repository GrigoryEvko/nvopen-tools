// Function: sub_198F1E0
// Address: 0x198f1e0
//
__int64 __fastcall sub_198F1E0(__int64 *a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 *v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r12
  bool v9; // al
  __int64 v10; // r14
  __int64 v11; // r13
  __int64 v12; // rsi
  __int64 *v13; // r13
  __int64 *v14; // rdi
  char v15; // r9
  __int64 v16; // r8
  int v17; // edx
  unsigned int v18; // ecx
  __int64 *v19; // rax
  __int64 v20; // r10
  int v21; // ecx
  int v22; // edx
  unsigned int v23; // r12d
  unsigned int v24; // r11d
  __int64 *v25; // rax
  __int64 v26; // r10
  int v27; // edx
  unsigned int v28; // ecx
  __int64 *v29; // rax
  __int64 v30; // r11
  int v31; // r10d
  int v32; // edx
  unsigned int v33; // ecx
  __int64 *v34; // rax
  __int64 v35; // r11
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rdx
  int v42; // eax
  int v43; // eax
  __int64 v44; // rdx
  __int64 v45; // rdx
  int v46; // eax
  int v47; // eax
  int v48; // r10d
  __int64 *v49; // rax
  __int64 v50; // rbx
  __int64 v51; // r12
  __int64 *v52; // rbx
  __int64 v53; // rcx
  bool v54; // zf
  int v55; // r11d
  __int64 *v56; // [rsp+0h] [rbp-80h]
  __int64 v57; // [rsp+10h] [rbp-70h]
  __int64 *v58; // [rsp+18h] [rbp-68h]
  __int64 v59; // [rsp+20h] [rbp-60h]
  int v61; // [rsp+30h] [rbp-50h]
  char v62; // [rsp+34h] [rbp-4Ch]
  int v63; // [rsp+34h] [rbp-4Ch]
  __int64 *v64; // [rsp+38h] [rbp-48h]
  __int64 *v65; // [rsp+38h] [rbp-48h]
  __int64 v66[7]; // [rsp+48h] [rbp-38h] BYREF

  result = a2 - (char *)a1;
  v58 = (__int64 *)a2;
  v57 = a3;
  if ( a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v65 = (__int64 *)a2;
    goto LABEL_72;
  }
  v56 = a1 + 2;
  v59 = a4 + 16;
  while ( 2 )
  {
    v66[0] = a4;
    --v57;
    v6 = &a1[result >> 4];
    v7 = a1[1];
    v8 = *v6;
    v64 = v6;
    v9 = sub_198ECB0(v66, v7, *v6);
    v10 = *a1;
    v11 = *(v58 - 1);
    if ( !v9 )
    {
      if ( !sub_198ECB0(v66, v7, v11) )
      {
        v54 = !sub_198ECB0(v66, v8, v11);
        v49 = a1;
        if ( v54 )
        {
          *a1 = v8;
          *v64 = v10;
          v7 = *a1;
          v10 = a1[1];
          v12 = *(v58 - 1);
          goto LABEL_7;
        }
        goto LABEL_78;
      }
      v49 = a1;
LABEL_70:
      *v49 = v7;
      v49[1] = v10;
      v12 = *(v58 - 1);
      goto LABEL_7;
    }
    if ( !sub_198ECB0(v66, v8, v11) )
    {
      v54 = !sub_198ECB0(v66, v7, v11);
      v49 = a1;
      if ( !v54 )
      {
LABEL_78:
        *v49 = v11;
        v12 = v10;
        *(v58 - 1) = v10;
        v7 = *v49;
        v10 = v49[1];
        goto LABEL_7;
      }
      goto LABEL_70;
    }
    *a1 = v8;
    *v64 = v10;
    v12 = *(v58 - 1);
    v7 = *a1;
    v10 = a1[1];
LABEL_7:
    v13 = v56;
    v14 = v58;
    v62 = *(_BYTE *)(a4 + 8);
    while ( 1 )
    {
      v65 = v13 - 1;
      v15 = v62 & 1;
      if ( (v62 & 1) != 0 )
      {
        v16 = v59;
        v17 = 15;
      }
      else
      {
        v41 = *(unsigned int *)(a4 + 24);
        v16 = *(_QWORD *)(a4 + 16);
        if ( !(_DWORD)v41 )
          goto LABEL_51;
        v17 = v41 - 1;
      }
      v18 = v17 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v19 = (__int64 *)(v16 + 16LL * v18);
      v20 = *v19;
      if ( *v19 == v10 )
        goto LABEL_11;
      v46 = 1;
      while ( v20 != -8 )
      {
        v55 = v46 + 1;
        v18 = v17 & (v46 + v18);
        v19 = (__int64 *)(v16 + 16LL * v18);
        v20 = *v19;
        if ( *v19 == v10 )
          goto LABEL_11;
        v46 = v55;
      }
      if ( v15 )
      {
        v45 = 256;
        goto LABEL_52;
      }
      v41 = *(unsigned int *)(a4 + 24);
LABEL_51:
      v45 = 16 * v41;
LABEL_52:
      v19 = (__int64 *)(v16 + v45);
LABEL_11:
      v21 = *((_DWORD *)v19 + 2);
      if ( v15 )
      {
        v22 = 15;
      }
      else
      {
        v40 = *(unsigned int *)(a4 + 24);
        if ( !(_DWORD)v40 )
          goto LABEL_48;
        v22 = v40 - 1;
      }
      v23 = ((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4);
      v24 = v22 & v23;
      v25 = (__int64 *)(v16 + 16LL * (v22 & v23));
      v26 = *v25;
      if ( *v25 == v7 )
        goto LABEL_14;
      v47 = 1;
      while ( v26 != -8 )
      {
        v24 = v22 & (v47 + v24);
        v61 = v47 + 1;
        v25 = (__int64 *)(v16 + 16LL * v24);
        v26 = *v25;
        if ( *v25 == v7 )
          goto LABEL_14;
        v47 = v61;
      }
      if ( v15 )
      {
        v44 = 256;
        goto LABEL_49;
      }
      v40 = *(unsigned int *)(a4 + 24);
LABEL_48:
      v44 = 16 * v40;
      v23 = ((unsigned int)v7 >> 4) ^ ((unsigned int)v7 >> 9);
LABEL_49:
      v25 = (__int64 *)(v16 + v44);
LABEL_14:
      if ( v21 >= *((_DWORD *)v25 + 2) )
        break;
LABEL_34:
      v10 = *v13++;
    }
    for ( --v14; ; --v14 )
    {
      if ( v15 )
      {
        v27 = 15;
      }
      else
      {
        v36 = *(unsigned int *)(a4 + 24);
        v27 = v36 - 1;
        if ( !(_DWORD)v36 )
          goto LABEL_25;
      }
      v28 = v23 & v27;
      v29 = (__int64 *)(v16 + 16LL * (v23 & v27));
      v30 = *v29;
      if ( *v29 == v7 )
      {
LABEL_18:
        v31 = *((_DWORD *)v29 + 2);
        if ( v15 )
          goto LABEL_19;
        goto LABEL_27;
      }
      v42 = 1;
      while ( v30 != -8 )
      {
        v48 = v42 + 1;
        v28 = v27 & (v42 + v28);
        v29 = (__int64 *)(v16 + 16LL * v28);
        v30 = *v29;
        if ( *v29 == v7 )
          goto LABEL_18;
        v42 = v48;
      }
      if ( !v15 )
      {
        v36 = *(unsigned int *)(a4 + 24);
LABEL_25:
        v37 = 16 * v36;
        goto LABEL_26;
      }
      v37 = 256;
LABEL_26:
      v31 = *(_DWORD *)(v16 + v37 + 8);
      if ( v15 )
      {
LABEL_19:
        v32 = 15;
        goto LABEL_20;
      }
LABEL_27:
      v38 = *(unsigned int *)(a4 + 24);
      if ( !(_DWORD)v38 )
        goto LABEL_30;
      v32 = v38 - 1;
LABEL_20:
      v33 = v32 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v34 = (__int64 *)(v16 + 16LL * v33);
      v35 = *v34;
      if ( v12 != *v34 )
        break;
LABEL_21:
      if ( v31 >= *((_DWORD *)v34 + 2) )
        goto LABEL_32;
LABEL_22:
      v12 = *(v14 - 1);
    }
    v43 = 1;
    while ( v35 != -8 )
    {
      v33 = v32 & (v43 + v33);
      v63 = v43 + 1;
      v34 = (__int64 *)(v16 + 16LL * v33);
      v35 = *v34;
      if ( v12 == *v34 )
        goto LABEL_21;
      v43 = v63;
    }
    if ( !v15 )
    {
      v38 = *(unsigned int *)(a4 + 24);
LABEL_30:
      v39 = 16 * v38;
      goto LABEL_31;
    }
    v39 = 256;
LABEL_31:
    if ( v31 < *(_DWORD *)(v16 + v39 + 8) )
      goto LABEL_22;
LABEL_32:
    if ( v14 > v65 )
    {
      *(v13 - 1) = v12;
      *v14 = v10;
      v12 = *(v14 - 1);
      v7 = *a1;
      v62 = *(_BYTE *)(a4 + 8);
      goto LABEL_34;
    }
    sub_198F1E0(v65, v58, v57, a4);
    result = (char *)v65 - (char *)a1;
    if ( (char *)v65 - (char *)a1 > 128 )
    {
      if ( v57 )
      {
        v58 = v13 - 1;
        continue;
      }
LABEL_72:
      v50 = result >> 3;
      v51 = ((result >> 3) - 2) >> 1;
      sub_198EE20((__int64)a1, v51, result >> 3, a1[v51], a4);
      do
      {
        --v51;
        sub_198EE20((__int64)a1, v51, v50, a1[v51], a4);
      }
      while ( v51 );
      v52 = v65;
      do
      {
        v53 = *--v52;
        *v52 = *a1;
        result = sub_198EE20((__int64)a1, 0, v52 - a1, v53, a4);
      }
      while ( (char *)v52 - (char *)a1 > 8 );
    }
    return result;
  }
}
