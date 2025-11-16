// Function: sub_1ACFF60
// Address: 0x1acff60
//
char __fastcall sub_1ACFF60(__int64 a1, __int64 a2, __int64 a3, size_t *a4)
{
  __int64 v4; // r10
  size_t *v5; // r11
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // r10
  size_t **v10; // r8
  size_t *v11; // r15
  __int64 v12; // rbx
  __int64 v13; // r9
  size_t **v14; // r12
  size_t *v15; // rcx
  size_t v16; // rdx
  size_t v17; // rax
  int v18; // edi
  bool v19; // zf
  bool v20; // sf
  bool v21; // of
  int v22; // eax
  size_t v23; // r11
  const void *v24; // rsi
  const void *v25; // rdi
  int v26; // eax
  size_t **v27; // r14
  __int64 v28; // r12
  const void *v29; // rcx
  size_t *v30; // r15
  int v31; // esi
  size_t v32; // rdx
  bool v33; // zf
  bool v34; // sf
  bool v35; // of
  size_t v36; // r9
  size_t v37; // r8
  const void *v38; // rdi
  size_t **v40; // [rsp+0h] [rbp-80h]
  size_t **v41; // [rsp+0h] [rbp-80h]
  __int64 v42; // [rsp+8h] [rbp-78h]
  __int64 v43; // [rsp+8h] [rbp-78h]
  size_t *v44; // [rsp+10h] [rbp-70h]
  size_t v45; // [rsp+10h] [rbp-70h]
  size_t *v46; // [rsp+18h] [rbp-68h]
  size_t v47; // [rsp+20h] [rbp-60h]
  size_t *v49; // [rsp+28h] [rbp-58h]
  size_t *v50; // [rsp+28h] [rbp-58h]
  __int64 v52; // [rsp+30h] [rbp-50h]
  __int64 v53; // [rsp+30h] [rbp-50h]
  size_t v55; // [rsp+38h] [rbp-48h]
  size_t v56; // [rsp+38h] [rbp-48h]
  __int64 v57; // [rsp+40h] [rbp-40h]
  size_t v58; // [rsp+40h] [rbp-40h]
  size_t v59; // [rsp+40h] [rbp-40h]
  size_t n; // [rsp+48h] [rbp-38h]
  size_t na; // [rsp+48h] [rbp-38h]
  size_t nb; // [rsp+48h] [rbp-38h]

  v4 = a2;
  v5 = a4;
  v57 = a3 & 1;
  v7 = (a3 - 1) / 2;
  if ( a2 < v7 )
  {
    v8 = a2;
    v9 = (a3 - 1) / 2;
    while ( 1 )
    {
      v13 = 2 * (v8 + 1);
      v12 = v13 - 1;
      v14 = (size_t **)(a1 + 16 * (v8 + 1));
      v10 = (size_t **)(a1 + 8 * (v13 - 1));
      v15 = *v14;
      v11 = *v10;
      v16 = (*v14)[1];
      v17 = (*v10)[1];
      v18 = *(_DWORD *)(v17 + 80);
      v21 = __OFSUB__(*(_DWORD *)(v16 + 80), v18);
      v19 = *(_DWORD *)(v16 + 80) == v18;
      v20 = *(_DWORD *)(v16 + 80) - v18 < 0;
      if ( *(_DWORD *)(v16 + 80) != v18 )
        break;
      v22 = *(_DWORD *)(v17 + 84);
      v21 = __OFSUB__(*(_DWORD *)(v16 + 84), v22);
      v19 = *(_DWORD *)(v16 + 84) == v22;
      v20 = *(_DWORD *)(v16 + 84) - v22 < 0;
      if ( *(_DWORD *)(v16 + 84) != v22 )
        break;
      v23 = *v15;
      v24 = v11 + 2;
      v25 = v15 + 2;
      n = *v11;
      if ( *v11 < *v15 )
      {
        if ( !n )
          goto LABEL_4;
        v41 = (size_t **)(a1 + 8 * (v13 - 1));
        v43 = v9;
        v45 = *v15;
        v46 = *v14;
        v26 = memcmp(v25, v24, n);
        v13 = 2 * (v8 + 1);
        v15 = v46;
        v23 = v45;
        v9 = v43;
        v10 = v41;
        if ( v26 )
        {
LABEL_40:
          if ( v26 < 0 )
          {
LABEL_5:
            *(_QWORD *)(a1 + 8 * v8) = v11;
            if ( v12 >= v9 )
              goto LABEL_15;
            goto LABEL_6;
          }
LABEL_4:
          v10 = (size_t **)(a1 + 16 * (v8 + 1));
          v11 = v15;
          v12 = v13;
          goto LABEL_5;
        }
      }
      else
      {
        if ( v23 )
        {
          v40 = (size_t **)(a1 + 8 * (v13 - 1));
          v42 = v9;
          v44 = *v14;
          v47 = *v15;
          v26 = memcmp(v25, v24, v23);
          v23 = v47;
          v13 = 2 * (v8 + 1);
          v15 = v44;
          v9 = v42;
          v10 = v40;
          if ( v26 )
            goto LABEL_40;
        }
        if ( n == v23 )
          goto LABEL_4;
      }
      if ( n <= v23 )
        goto LABEL_4;
      *(_QWORD *)(a1 + 8 * v8) = v11;
      if ( v12 >= v9 )
      {
LABEL_15:
        v4 = a2;
        v27 = v10;
        v5 = a4;
        if ( v57 )
          goto LABEL_16;
        goto LABEL_24;
      }
LABEL_6:
      v8 = v12;
    }
    if ( !(v20 ^ v21 | v19) )
      goto LABEL_5;
    goto LABEL_4;
  }
  v27 = (size_t **)(a1 + 8 * a2);
  if ( (a3 & 1) == 0 )
  {
    v12 = a2;
LABEL_24:
    if ( (a3 - 2) / 2 == v12 )
    {
      v12 = 2 * v12 + 1;
      *v27 = *(size_t **)(a1 + 8 * v12);
      v27 = (size_t **)(a1 + 8 * v12);
    }
LABEL_16:
    LOBYTE(v7) = v12 - 1;
    v28 = (v12 - 1) / 2;
    if ( v12 > v4 )
    {
      v29 = v5 + 2;
      while ( 1 )
      {
        v27 = (size_t **)(a1 + 8 * v28);
        v7 = v5[1];
        v30 = *v27;
        v31 = *(_DWORD *)(v7 + 80);
        v32 = (*v27)[1];
        v35 = __OFSUB__(*(_DWORD *)(v32 + 80), v31);
        v33 = *(_DWORD *)(v32 + 80) == v31;
        v34 = *(_DWORD *)(v32 + 80) - v31 < 0;
        if ( *(_DWORD *)(v32 + 80) != v31
          || (LODWORD(v7) = *(_DWORD *)(v7 + 84),
              v35 = __OFSUB__(*(_DWORD *)(v32 + 84), (_DWORD)v7),
              v33 = *(_DWORD *)(v32 + 84) == (_DWORD)v7,
              v34 = *(_DWORD *)(v32 + 84) - (int)v7 < 0,
              *(_DWORD *)(v32 + 84) != (_DWORD)v7) )
        {
          LOBYTE(v7) = !(v34 ^ v35 | v33);
          if ( v34 ^ v35 | v33 )
            goto LABEL_32;
          goto LABEL_20;
        }
        v36 = *v5;
        v37 = *v30;
        v38 = v30 + 2;
        if ( *v5 < *v30 )
        {
          if ( !v36 )
            goto LABEL_32;
          v50 = v5;
          v53 = v4;
          v56 = *v30;
          v59 = *v5;
          nb = (size_t)v29;
          LODWORD(v7) = memcmp(v38, v29, *v5);
          v29 = (const void *)nb;
          v36 = v59;
          v37 = v56;
          v4 = v53;
          v5 = v50;
          if ( !(_DWORD)v7 )
            goto LABEL_31;
        }
        else
        {
          if ( !v37 )
            goto LABEL_30;
          v49 = v5;
          v52 = v4;
          v55 = *v5;
          v58 = *v30;
          na = (size_t)v29;
          LODWORD(v7) = memcmp(v38, v29, *v30);
          v29 = (const void *)na;
          v37 = v58;
          v36 = v55;
          v4 = v52;
          v5 = v49;
          if ( !(_DWORD)v7 )
          {
LABEL_30:
            if ( v36 == v37 )
              goto LABEL_32;
LABEL_31:
            if ( v36 <= v37 )
              goto LABEL_32;
            goto LABEL_20;
          }
        }
        if ( (int)v7 >= 0 )
        {
LABEL_32:
          v27 = (size_t **)(a1 + 8 * v12);
          break;
        }
LABEL_20:
        *(_QWORD *)(a1 + 8 * v12) = v30;
        v12 = v28;
        v7 = (v28 - 1) / 2;
        if ( v4 >= v28 )
          break;
        v28 = (v28 - 1) / 2;
      }
    }
  }
  *v27 = v5;
  return v7;
}
