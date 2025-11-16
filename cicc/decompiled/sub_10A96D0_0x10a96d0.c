// Function: sub_10A96D0
// Address: 0x10a96d0
//
__int64 __fastcall sub_10A96D0(__int64 a1, int a2, unsigned __int8 *a3)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rax
  int v9; // r13d
  __int64 v10; // r15
  void *v11; // rax
  unsigned __int8 *v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r15
  void *v19; // rax
  __int64 v20; // rax
  bool v21; // al
  __int64 *v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r14
  _BYTE *v25; // rax
  _BYTE *v26; // r13
  void *v27; // rax
  _BYTE *v28; // r13
  __int64 v29; // r14
  _BYTE *v30; // rax
  _BYTE *v31; // r13
  void *v32; // rax
  _BYTE *v33; // r13
  int v34; // eax
  char v35; // cl
  unsigned int v36; // r14d
  void **v37; // rax
  void **v38; // r13
  char v39; // al
  void *v40; // rax
  _BYTE *v41; // r13
  int v42; // eax
  char v43; // cl
  unsigned int v44; // r14d
  void **v45; // rax
  void **v46; // r13
  char v47; // al
  void *v48; // rax
  _BYTE *v49; // r13
  unsigned __int8 *v50; // [rsp-50h] [rbp-50h]
  unsigned __int8 *v51; // [rsp-50h] [rbp-50h]
  char v52; // [rsp-48h] [rbp-48h]
  char v53; // [rsp-48h] [rbp-48h]
  unsigned __int8 *v54; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v55; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v56; // [rsp-40h] [rbp-40h]
  int v57; // [rsp-40h] [rbp-40h]
  int v58; // [rsp-40h] [rbp-40h]

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  v5 = *(_QWORD *)(v4 + 16);
  if ( !v5 )
    goto LABEL_4;
  if ( *(_QWORD *)(v5 + 8) )
    goto LABEL_4;
  if ( *(_BYTE *)v4 != 85 )
    goto LABEL_4;
  v17 = *(_QWORD *)(v4 - 32);
  if ( !v17 || *(_BYTE *)v17 || *(_QWORD *)(v17 + 24) != *(_QWORD *)(v4 + 80) || *(_DWORD *)(v17 + 36) != *(_DWORD *)a1 )
    goto LABEL_4;
  v18 = *(_QWORD *)(v4 + 32 * (*(unsigned int *)(a1 + 8) - (unsigned __int64)(*(_DWORD *)(v4 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)v18 == 18 )
  {
    v55 = a3;
    v19 = sub_C33340();
    a3 = v55;
    if ( *(void **)(v18 + 24) == v19 )
      v20 = *(_QWORD *)(v18 + 32);
    else
      v20 = v18 + 24;
    v21 = (*(_BYTE *)(v20 + 20) & 7) == 3;
    goto LABEL_34;
  }
  v24 = *(_QWORD *)(v18 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17 > 1 || *(_BYTE *)v18 > 0x15u )
    goto LABEL_4;
  v56 = a3;
  v25 = sub_AD7630(v18, 0, (__int64)a3);
  a3 = v56;
  v26 = v25;
  if ( v25 && *v25 == 18 )
  {
    v27 = sub_C33340();
    a3 = v56;
    if ( *((void **)v26 + 3) == v27 )
      v28 = (_BYTE *)*((_QWORD *)v26 + 4);
    else
      v28 = v26 + 24;
    v21 = (v28[20] & 7) == 3;
LABEL_34:
    if ( v21 )
      goto LABEL_35;
LABEL_4:
    v6 = *((_QWORD *)a3 - 4);
LABEL_5:
    v7 = *(_QWORD *)(v6 + 16);
    if ( !v7 )
      return 0;
    if ( *(_QWORD *)(v7 + 8) )
      return 0;
    if ( *(_BYTE *)v6 != 85 )
      return 0;
    v8 = *(_QWORD *)(v6 - 32);
    if ( !v8 || *(_BYTE *)v8 || *(_QWORD *)(v8 + 24) != *(_QWORD *)(v6 + 80) || *(_DWORD *)(v8 + 36) != *(_DWORD *)a1 )
      return 0;
    v9 = *(_DWORD *)(v6 + 4);
    v54 = a3;
    v10 = *(_QWORD *)(v6 + 32 * (*(unsigned int *)(a1 + 8) - (unsigned __int64)(v9 & 0x7FFFFFF)));
    if ( *(_BYTE *)v10 == 18 )
    {
      v11 = sub_C33340();
      v12 = v54;
      if ( *(void **)(v10 + 24) == v11 )
        v13 = *(_QWORD *)(v10 + 32);
      else
        v13 = v10 + 24;
      if ( (*(_BYTE *)(v13 + 20) & 7) != 3 )
        return 0;
      v14 = *(__int64 **)(a1 + 16);
      if ( !v14 )
        goto LABEL_21;
    }
    else
    {
      v29 = *(_QWORD *)(v10 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v29 + 8) - 17 > 1 || *(_BYTE *)v10 > 0x15u )
        return 0;
      v30 = sub_AD7630(v10, 0, (__int64)a3);
      v12 = v54;
      v31 = v30;
      if ( v30 && *v30 == 18 )
      {
        v32 = sub_C33340();
        v12 = v54;
        if ( *((void **)v31 + 3) == v32 )
          v33 = (_BYTE *)*((_QWORD *)v31 + 4);
        else
          v33 = v31 + 24;
        if ( (v33[20] & 7) != 3 )
          return 0;
      }
      else
      {
        if ( *(_BYTE *)(v29 + 8) != 17 )
          return 0;
        v42 = *(_DWORD *)(v29 + 32);
        v43 = 0;
        v44 = 0;
        v58 = v42;
        while ( v58 != v44 )
        {
          v51 = v12;
          v53 = v43;
          v45 = (void **)sub_AD69F0((unsigned __int8 *)v10, v44);
          v46 = v45;
          if ( !v45 )
            return 0;
          v47 = *(_BYTE *)v45;
          v43 = v53;
          v12 = v51;
          if ( v47 != 13 )
          {
            if ( v47 != 18 )
              return 0;
            v48 = sub_C33340();
            v12 = v51;
            v49 = v46[3] == v48 ? v46[4] : v46 + 3;
            if ( (v49[20] & 7) != 3 )
              return 0;
            v43 = 1;
          }
          ++v44;
        }
        if ( !v43 )
          return 0;
      }
      v14 = *(__int64 **)(a1 + 16);
      if ( !v14 )
        goto LABEL_19;
    }
    *v14 = v10;
LABEL_19:
    if ( *(_BYTE *)v6 != 85 )
      return 0;
    v9 = *(_DWORD *)(v6 + 4);
LABEL_21:
    v15 = *(_QWORD *)(v6 + 32 * (*(unsigned int *)(a1 + 24) - (unsigned __int64)(v9 & 0x7FFFFFF)));
    if ( v15 )
    {
      **(_QWORD **)(a1 + 32) = v15;
      v16 = *((_QWORD *)v12 - 8);
      if ( v16 )
      {
        **(_QWORD **)(a1 + 40) = v16;
        return 1;
      }
    }
    return 0;
  }
  if ( *(_BYTE *)(v24 + 8) != 17 )
    goto LABEL_4;
  v34 = *(_DWORD *)(v24 + 32);
  v35 = 0;
  v36 = 0;
  v57 = v34;
  while ( v57 != v36 )
  {
    v50 = a3;
    v52 = v35;
    v37 = (void **)sub_AD69F0((unsigned __int8 *)v18, v36);
    v35 = v52;
    a3 = v50;
    v38 = v37;
    if ( !v37 )
      goto LABEL_4;
    v39 = *(_BYTE *)v37;
    if ( v39 != 13 )
    {
      if ( v39 != 18 )
        goto LABEL_4;
      v40 = sub_C33340();
      a3 = v50;
      v41 = v38[3] == v40 ? v38[4] : v38 + 3;
      if ( (v41[20] & 7) != 3 )
        goto LABEL_4;
      v35 = 1;
    }
    ++v36;
  }
  if ( !v35 )
    goto LABEL_4;
LABEL_35:
  v22 = *(__int64 **)(a1 + 16);
  if ( v22 )
    *v22 = v18;
  if ( *(_BYTE *)v4 != 85 )
    goto LABEL_4;
  v23 = *(_QWORD *)(v4 + 32 * (*(unsigned int *)(a1 + 24) - (unsigned __int64)(*(_DWORD *)(v4 + 4) & 0x7FFFFFF)));
  if ( !v23 )
    goto LABEL_4;
  **(_QWORD **)(a1 + 32) = v23;
  v6 = *((_QWORD *)a3 - 4);
  if ( !v6 )
    goto LABEL_5;
  **(_QWORD **)(a1 + 40) = v6;
  return 1;
}
