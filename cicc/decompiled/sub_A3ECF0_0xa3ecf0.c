// Function: sub_A3ECF0
// Address: 0xa3ecf0
//
signed __int64 __fastcall sub_A3ECF0(_DWORD *a1, unsigned __int64 *a2, __int64 a3, __int64 a4)
{
  signed __int64 result; // rax
  unsigned int v5; // r9d
  unsigned int *v6; // rcx
  unsigned int v7; // r12d
  unsigned int v8; // r15d
  unsigned int v9; // r14d
  __int64 v10; // r8
  unsigned int v11; // r13d
  char *v12; // rsi
  char v13; // al
  unsigned int v14; // r10d
  char *v15; // rdi
  char v16; // dl
  unsigned int v17; // ebx
  char *v18; // r10
  unsigned int v19; // r9d
  char v20; // r8
  unsigned int v21; // r11d
  unsigned int v22; // eax
  _DWORD *v23; // rbx
  __int64 v24; // rax
  unsigned int v25; // r14d
  unsigned int v26; // edi
  unsigned int v27; // r8d
  unsigned int v28; // ecx
  unsigned int v29; // esi
  unsigned int *v30; // rbx
  __int64 v31; // r10
  unsigned int v32; // r11d
  char *v33; // r13
  unsigned __int64 *v34; // rax
  char v35; // r9
  unsigned int v36; // esi
  unsigned __int64 *v37; // r15
  unsigned int v38; // ecx
  unsigned int v39; // esi
  _BYTE *v40; // r12
  unsigned int v41; // ecx
  _BYTE *v42; // rsi
  unsigned int v43; // edx
  unsigned int v44; // ebx
  char *v45; // r11
  unsigned int v46; // r9d
  char v47; // r8
  unsigned int v48; // r10d
  unsigned int v49; // edi
  unsigned int v50; // edx
  unsigned int v51; // edi
  __int64 v52; // rbx
  __int64 v53; // r12
  __int64 v54; // r8
  unsigned __int64 v55; // rcx
  unsigned int v56; // esi
  unsigned int *v57; // [rsp+0h] [rbp-60h]
  __int64 v58; // [rsp+8h] [rbp-58h]
  unsigned __int64 *v59; // [rsp+10h] [rbp-50h]
  unsigned int v62; // [rsp+28h] [rbp-38h]
  unsigned int v63; // [rsp+2Ch] [rbp-34h]
  unsigned int v64; // [rsp+2Ch] [rbp-34h]

  result = (char *)a2 - (char *)a1;
  v59 = a2;
  v58 = a3;
  if ( (char *)a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v37 = a2;
    goto LABEL_92;
  }
  v57 = a1 + 4;
  while ( 2 )
  {
    --v58;
    v5 = 0;
    v6 = &a1[2 * (result >> 4)];
    v7 = a1[2];
    v8 = v6[1];
    v9 = a1[3];
    v10 = *(_QWORD *)(a4 + 208);
    v11 = *v6;
    v12 = *(char **)(v10 + 8LL * (v8 - 1));
    v13 = *v12;
    if ( *v12 )
    {
      v5 = 1;
      if ( (unsigned __int8)(v13 - 5) <= 0x1Fu )
        v5 = ((v12[1] & 0x7F) != 1) + 2;
    }
    v14 = 0;
    v15 = *(char **)(v10 + 8LL * (v9 - 1));
    v16 = *v15;
    if ( *v15 )
    {
      v14 = 1;
      if ( (unsigned __int8)(v16 - 5) <= 0x1Fu )
        v14 = ((v15[1] & 0x7F) != 1) + 2;
    }
    if ( v7 >= v11 && (v7 != v11 || v14 >= v5 && (v14 != v5 || v9 >= v8)) )
    {
      v44 = *((_DWORD *)v59 - 1);
      v64 = *((_DWORD *)v59 - 2);
      v45 = *(char **)(v10 + 8LL * (v44 - 1));
      v46 = 0;
      v47 = *v45;
      if ( *v45 )
      {
        v46 = 1;
        if ( (unsigned __int8)(v47 - 5) <= 0x1Fu )
          v46 = ((v45[1] & 0x7F) != 1) + 2;
      }
      v48 = 0;
      if ( v16 )
      {
        v48 = 1;
        if ( (unsigned __int8)(v16 - 5) <= 0x1Fu )
          v48 = ((v15[1] & 0x7F) != 1) + 2;
      }
      if ( v7 < v64 || v7 == v64 && (v48 < v46 || v48 == v46 && v9 < v44) )
      {
        v49 = *a1;
        v25 = a1[1];
        *(_QWORD *)a1 = *((_QWORD *)a1 + 1);
        a1[2] = v49;
        a1[3] = v25;
        v27 = *((_DWORD *)v59 - 2);
        v63 = v49;
        v26 = *((_DWORD *)v59 - 1);
        goto LABEL_34;
      }
      v50 = 0;
      if ( v47 )
      {
        v50 = 1;
        if ( (unsigned __int8)(v47 - 5) <= 0x1Fu )
          v50 = ((v45[1] & 0x7F) != 1) + 2;
      }
      v51 = 0;
      if ( v13 )
      {
        v51 = 1;
        if ( (unsigned __int8)(v13 - 5) <= 0x1Fu )
          v51 = ((v12[1] & 0x7F) != 1) + 2;
      }
      if ( v11 >= v64 && (v11 != v64 || v51 >= v50 && (v51 != v50 || v44 <= v8)) )
      {
        v23 = a1;
        v24 = *(_QWORD *)a1;
        goto LABEL_25;
      }
LABEL_33:
      v27 = *a1;
      v26 = a1[1];
      *(_QWORD *)a1 = *(v59 - 1);
      *((_DWORD *)v59 - 2) = v27;
      *((_DWORD *)v59 - 1) = v26;
      v25 = a1[3];
      v63 = a1[2];
      goto LABEL_34;
    }
    v17 = *((_DWORD *)v59 - 1);
    v18 = *(char **)(v10 + 8LL * (v17 - 1));
    v19 = 0;
    v20 = *v18;
    if ( *v18 )
    {
      v19 = 1;
      if ( (unsigned __int8)(v20 - 5) <= 0x1Fu )
        v19 = ((v18[1] & 0x7F) != 1) + 2;
    }
    v21 = 0;
    if ( v13 )
    {
      v21 = 1;
      if ( (unsigned __int8)(v13 - 5) <= 0x1Fu )
        v21 = ((v12[1] & 0x7F) != 1) + 2;
    }
    v22 = *((_DWORD *)v59 - 2);
    if ( v11 >= v22 && (v11 != v22 || v21 >= v19 && (v21 != v19 || v17 <= v8)) )
    {
      v28 = 0;
      if ( v20 )
      {
        v28 = 1;
        if ( (unsigned __int8)(v20 - 5) <= 0x1Fu )
          v28 = ((v18[1] & 0x7F) != 1) + 2;
      }
      v29 = 0;
      if ( v16 )
      {
        v29 = 1;
        if ( (unsigned __int8)(v16 - 5) <= 0x1Fu )
          v29 = ((v15[1] & 0x7F) != 1) + 2;
      }
      if ( v7 >= v22 && (v7 != v22 || v29 >= v28 && (v29 != v28 || v9 >= v17)) )
      {
        v56 = *a1;
        v25 = a1[1];
        *(_QWORD *)a1 = *((_QWORD *)a1 + 1);
        a1[2] = v56;
        a1[3] = v25;
        v26 = *((_DWORD *)v59 - 1);
        v63 = v56;
        v27 = *((_DWORD *)v59 - 2);
        goto LABEL_34;
      }
      goto LABEL_33;
    }
    v23 = a1;
    v24 = *(_QWORD *)a1;
LABEL_25:
    *(_QWORD *)v23 = *(_QWORD *)v6;
    *(_QWORD *)v6 = v24;
    v25 = v23[3];
    v63 = v23[2];
    v26 = *((_DWORD *)v59 - 1);
    v27 = *((_DWORD *)v59 - 2);
LABEL_34:
    v30 = v57;
    v31 = *(_QWORD *)(a4 + 208);
    v32 = *a1;
    v62 = a1[1];
    v33 = *(char **)(v31 + 8LL * (v62 - 1));
    v34 = v59;
    v35 = *v33;
    while ( 1 )
    {
      v37 = (unsigned __int64 *)(v30 - 2);
      v38 = 0;
      if ( v35 )
      {
        v38 = 1;
        if ( (unsigned __int8)(v35 - 5) <= 0x1Fu )
          v38 = ((v33[1] & 0x7F) != 1) + 2;
      }
      v39 = 0;
      v40 = *(_BYTE **)(v31 + 8LL * (v25 - 1));
      if ( *v40 )
      {
        v39 = 1;
        if ( (unsigned __int8)(*v40 - 5) <= 0x1Fu )
          v39 = ((v40[1] & 0x7F) != 1) + 2;
      }
      if ( v63 >= v32 && (v63 != v32 || v39 >= v38 && (v39 != v38 || v25 >= v62)) )
        break;
LABEL_41:
      v36 = *v30;
      v25 = v30[1];
      v30 += 2;
      v63 = v36;
    }
    for ( --v34; ; v27 = *(_DWORD *)v34 )
    {
      v41 = 0;
      v42 = *(_BYTE **)(v31 + 8LL * (v26 - 1));
      if ( *v42 )
      {
        v41 = 1;
        if ( (unsigned __int8)(*v42 - 5) <= 0x1Fu )
          v41 = ((v42[1] & 0x7F) != 1) + 2;
      }
      v43 = 0;
      if ( v35 )
      {
        v43 = 1;
        if ( (unsigned __int8)(v35 - 5) <= 0x1Fu )
          v43 = ((v33[1] & 0x7F) != 1) + 2;
      }
      if ( v27 <= v32 && (v27 != v32 || v43 >= v41 && (v43 != v41 || v26 <= v62)) )
        break;
      v26 = *((_DWORD *)--v34 + 1);
    }
    if ( v34 > v37 )
    {
      *((_QWORD *)v30 - 1) = *v34;
      v27 = *((_DWORD *)v34 - 2);
      *(_DWORD *)v34 = v63;
      *((_DWORD *)v34 + 1) = v25;
      v31 = *(_QWORD *)(a4 + 208);
      v32 = *a1;
      v26 = *((_DWORD *)v34 - 1);
      v62 = a1[1];
      v33 = *(char **)(v31 + 8LL * (v62 - 1));
      v35 = *v33;
      goto LABEL_41;
    }
    sub_A3ECF0(v30 - 2, v59, v58, a4);
    result = (char *)v37 - (char *)a1;
    if ( (char *)v37 - (char *)a1 > 128 )
    {
      if ( v58 )
      {
        v59 = (unsigned __int64 *)(v30 - 2);
        continue;
      }
LABEL_92:
      v52 = result >> 3;
      v53 = ((result >> 3) - 2) >> 1;
      sub_A3E750((__int64)a1, v53, result >> 3, *(_QWORD *)&a1[2 * v53], a4);
      do
      {
        --v53;
        sub_A3E750((__int64)a1, v53, v52, *(_QWORD *)&a1[2 * v53], v54);
      }
      while ( v53 );
      do
      {
        v55 = *--v37;
        *v37 = *(_QWORD *)a1;
        result = (signed __int64)sub_A3E750((__int64)a1, 0, ((char *)v37 - (char *)a1) >> 3, v55, v54);
      }
      while ( (char *)v37 - (char *)a1 > 8 );
    }
    return result;
  }
}
