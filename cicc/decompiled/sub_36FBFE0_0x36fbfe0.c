// Function: sub_36FBFE0
// Address: 0x36fbfe0
//
char __fastcall sub_36FBFE0(__int64 a1, __int64 a2, __int64 a3, size_t *a4)
{
  __int64 v4; // r11
  size_t *v5; // r10
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
  size_t v24; // rdx
  int v25; // eax
  size_t **v26; // r14
  __int64 v27; // r12
  const void *v28; // rcx
  size_t *v29; // r15
  int v30; // esi
  size_t v31; // rdx
  bool v32; // zf
  bool v33; // sf
  bool v34; // of
  size_t v35; // r9
  size_t v36; // r8
  size_t v37; // rdx
  size_t **v39; // [rsp+0h] [rbp-80h]
  __int64 v40; // [rsp+8h] [rbp-78h]
  size_t v41; // [rsp+10h] [rbp-70h]
  size_t *v42; // [rsp+20h] [rbp-60h]
  size_t *v44; // [rsp+28h] [rbp-58h]
  size_t v45; // [rsp+30h] [rbp-50h]
  __int64 v46; // [rsp+30h] [rbp-50h]
  size_t v48; // [rsp+38h] [rbp-48h]
  size_t v49; // [rsp+40h] [rbp-40h]
  __int64 v50; // [rsp+48h] [rbp-38h]
  const void *v51; // [rsp+48h] [rbp-38h]

  v4 = a2;
  v5 = a4;
  v50 = a3 & 1;
  v7 = (a3 - 1) / 2;
  if ( a2 >= v7 )
  {
    v26 = (size_t **)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_35;
    v12 = a2;
LABEL_25:
    if ( (a3 - 2) / 2 == v12 )
    {
      v12 = 2 * v12 + 1;
      *v26 = *(size_t **)(a1 + 8 * v12);
      v26 = (size_t **)(a1 + 8 * v12);
    }
    goto LABEL_17;
  }
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
    if ( *(_DWORD *)(v16 + 80) != v18
      || (v22 = *(_DWORD *)(v17 + 84),
          v21 = __OFSUB__(*(_DWORD *)(v16 + 84), v22),
          v19 = *(_DWORD *)(v16 + 84) == v22,
          v20 = *(_DWORD *)(v16 + 84) - v22 < 0,
          *(_DWORD *)(v16 + 84) != v22) )
    {
      if ( !(v20 ^ v21 | v19) )
        goto LABEL_5;
      goto LABEL_4;
    }
    v23 = *v11;
    v24 = *v15;
    v45 = *v15;
    if ( *v11 <= *v15 )
      v24 = *v11;
    if ( !v24 )
      break;
    v39 = (size_t **)(a1 + 8 * (v13 - 1));
    v40 = v9;
    v41 = *v11;
    v42 = *v14;
    v25 = memcmp(v15 + 2, v11 + 2, v24);
    v15 = v42;
    v13 = 2 * (v8 + 1);
    v23 = v41;
    v9 = v40;
    v10 = v39;
    if ( !v25 )
      break;
    if ( v25 >= 0 )
      goto LABEL_4;
LABEL_5:
    *(_QWORD *)(a1 + 8 * v8) = v11;
    if ( v12 >= v9 )
      goto LABEL_16;
LABEL_6:
    v8 = v12;
  }
  if ( v23 == v45 || v23 <= v45 )
  {
LABEL_4:
    v10 = (size_t **)(a1 + 16 * (v8 + 1));
    v11 = v15;
    v12 = v13;
    goto LABEL_5;
  }
  *(_QWORD *)(a1 + 8 * v8) = v11;
  if ( v12 < v9 )
    goto LABEL_6;
LABEL_16:
  v4 = a2;
  v26 = v10;
  v5 = a4;
  if ( !v50 )
    goto LABEL_25;
LABEL_17:
  LOBYTE(v7) = v12 - 1;
  v27 = (v12 - 1) / 2;
  if ( v12 > v4 )
  {
    v28 = v5 + 2;
    while ( 1 )
    {
      v26 = (size_t **)(a1 + 8 * v27);
      v7 = v5[1];
      v29 = *v26;
      v30 = *(_DWORD *)(v7 + 80);
      v31 = (*v26)[1];
      v34 = __OFSUB__(*(_DWORD *)(v31 + 80), v30);
      v32 = *(_DWORD *)(v31 + 80) == v30;
      v33 = *(_DWORD *)(v31 + 80) - v30 < 0;
      if ( *(_DWORD *)(v31 + 80) == v30
        && (LODWORD(v7) = *(_DWORD *)(v7 + 84),
            v34 = __OFSUB__(*(_DWORD *)(v31 + 84), (_DWORD)v7),
            v32 = *(_DWORD *)(v31 + 84) == (_DWORD)v7,
            v33 = *(_DWORD *)(v31 + 84) - (int)v7 < 0,
            *(_DWORD *)(v31 + 84) == (_DWORD)v7) )
      {
        v35 = *v5;
        v36 = *v29;
        v37 = *v29;
        if ( *v5 <= *v29 )
          v37 = *v5;
        if ( !v37 )
          goto LABEL_33;
        v44 = v5;
        v46 = v4;
        v48 = *v29;
        v49 = *v5;
        v51 = v28;
        LODWORD(v7) = memcmp(v29 + 2, v28, v37);
        v28 = v51;
        v35 = v49;
        v36 = v48;
        v4 = v46;
        v5 = v44;
        if ( (_DWORD)v7 )
        {
          if ( (int)v7 >= 0 )
            goto LABEL_34;
        }
        else
        {
LABEL_33:
          if ( v35 == v36 || v35 <= v36 )
          {
LABEL_34:
            v26 = (size_t **)(a1 + 8 * v12);
            break;
          }
        }
      }
      else
      {
        LOBYTE(v7) = !(v33 ^ v34 | v32);
        if ( v33 ^ v34 | v32 )
          goto LABEL_34;
      }
      *(_QWORD *)(a1 + 8 * v12) = v29;
      v12 = v27;
      v7 = (v27 - 1) / 2;
      if ( v4 >= v27 )
        break;
      v27 = (v27 - 1) / 2;
    }
  }
LABEL_35:
  *v26 = v5;
  return v7;
}
