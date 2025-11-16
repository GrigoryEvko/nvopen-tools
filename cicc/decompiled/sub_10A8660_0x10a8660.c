// Function: sub_10A8660
// Address: 0x10a8660
//
__int64 __fastcall sub_10A8660(__int64 *a1, int a2, unsigned __int8 *a3)
{
  __int64 result; // rax
  _BYTE *v5; // rax
  __int64 v6; // rsi
  _BYTE *v7; // r13
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rcx
  __int64 v11; // r15
  unsigned int v12; // r12d
  __int64 v13; // rdi
  int v14; // eax
  bool v15; // al
  _QWORD *v16; // rax
  unsigned __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rax
  _BYTE *v20; // r13
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rcx
  __int64 v24; // r15
  unsigned int v25; // r12d
  int v26; // eax
  __int64 *v27; // rax
  unsigned __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // r12
  _BYTE *v33; // rax
  __int64 v34; // r12
  _BYTE *v35; // rax
  unsigned int v36; // r12d
  int v37; // eax
  bool v38; // al
  int v39; // eax
  unsigned int v40; // r14d
  bool v41; // r12
  __int64 v42; // rax
  unsigned int v43; // r12d
  int v44; // eax
  int v45; // eax
  unsigned int v46; // r14d
  bool v47; // r12
  __int64 v48; // rax
  unsigned int v49; // r12d
  int v50; // eax
  unsigned __int8 *v51; // [rsp-58h] [rbp-58h]
  unsigned __int8 *v52; // [rsp-58h] [rbp-58h]
  unsigned __int8 *v53; // [rsp-50h] [rbp-50h]
  unsigned __int8 *v54; // [rsp-50h] [rbp-50h]
  unsigned __int8 *v55; // [rsp-50h] [rbp-50h]
  unsigned __int8 *v56; // [rsp-50h] [rbp-50h]
  unsigned __int8 *v57; // [rsp-50h] [rbp-50h]
  unsigned __int8 *v58; // [rsp-50h] [rbp-50h]
  int v59; // [rsp-50h] [rbp-50h]
  int v60; // [rsp-50h] [rbp-50h]

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v5 != 68 )
    goto LABEL_4;
  v7 = (_BYTE *)*((_QWORD *)v5 - 4);
  if ( *v7 != 82 )
    goto LABEL_4;
  v8 = *((_QWORD *)v7 - 8);
  if ( *(_BYTE *)v8 != 85 )
    goto LABEL_4;
  v9 = *(_QWORD *)(v8 - 32);
  if ( !v9 )
    goto LABEL_4;
  if ( *(_BYTE *)v9 )
    goto LABEL_4;
  if ( *(_QWORD *)(v9 + 24) != *(_QWORD *)(v8 + 80) )
    goto LABEL_4;
  if ( *(_DWORD *)(v9 + 36) != *((_DWORD *)a1 + 2) )
    goto LABEL_4;
  v10 = *(_QWORD *)(v8 + 32 * (*((unsigned int *)a1 + 4) - (unsigned __int64)(*(_DWORD *)(v8 + 4) & 0x7FFFFFF)));
  if ( !v10 )
    goto LABEL_4;
  *(_QWORD *)a1[3] = v10;
  v11 = *((_QWORD *)v7 - 4);
  if ( *(_BYTE *)v11 == 17 )
  {
    v12 = *(_DWORD *)(v11 + 32);
    if ( v12 > 0x40 )
    {
      v53 = a3;
      v13 = v11 + 24;
LABEL_18:
      v14 = sub_C444A0(v13);
      a3 = v53;
      v15 = v12 - 1 == v14;
      goto LABEL_19;
    }
    v15 = *(_QWORD *)(v11 + 24) == 1;
  }
  else
  {
    v32 = *(_QWORD *)(v11 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v32 + 8) - 17 > 1 || *(_BYTE *)v11 > 0x15u )
      goto LABEL_4;
    v53 = a3;
    v33 = sub_AD7630(*((_QWORD *)v7 - 4), 0, (__int64)a3);
    a3 = v53;
    if ( !v33 || *v33 != 17 )
    {
      if ( *(_BYTE *)(v32 + 8) != 17 )
        goto LABEL_4;
      v39 = *(_DWORD *)(v32 + 32);
      v40 = 0;
      v41 = 0;
      v59 = v39;
      while ( v59 != v40 )
      {
        v51 = a3;
        v42 = sub_AD69F0((unsigned __int8 *)v11, v40);
        a3 = v51;
        if ( !v42 )
          goto LABEL_4;
        if ( *(_BYTE *)v42 != 13 )
        {
          if ( *(_BYTE *)v42 != 17 )
            goto LABEL_4;
          v43 = *(_DWORD *)(v42 + 32);
          if ( v43 <= 0x40 )
          {
            v41 = *(_QWORD *)(v42 + 24) == 1;
          }
          else
          {
            v44 = sub_C444A0(v42 + 24);
            a3 = v51;
            v41 = v43 - 1 == v44;
          }
          if ( !v41 )
            goto LABEL_4;
        }
        ++v40;
      }
      if ( !v41 )
        goto LABEL_4;
      goto LABEL_20;
    }
    v12 = *((_DWORD *)v33 + 8);
    if ( v12 > 0x40 )
    {
      v13 = (__int64)(v33 + 24);
      goto LABEL_18;
    }
    v15 = *((_QWORD *)v33 + 3) == 1;
  }
LABEL_19:
  if ( !v15 )
    goto LABEL_4;
LABEL_20:
  v16 = (_QWORD *)a1[4];
  if ( v16 )
    *v16 = v11;
  if ( *a1 )
  {
    v54 = a3;
    v17 = sub_B53900((__int64)v7);
    v18 = *a1;
    a3 = v54;
    *(_DWORD *)v18 = v17;
    *(_BYTE *)(v18 + 4) = BYTE4(v17);
  }
  v6 = *((_QWORD *)a3 - 4);
  v19 = *(_QWORD *)(v6 + 16);
  if ( !v19 || *(_QWORD *)(v19 + 8) )
  {
LABEL_5:
    if ( *(_BYTE *)v6 != 68 )
      return 0;
    v20 = *(_BYTE **)(v6 - 32);
    if ( *v20 != 82 )
      return 0;
    v21 = *((_QWORD *)v20 - 8);
    if ( *(_BYTE *)v21 != 85 )
      return 0;
    v22 = *(_QWORD *)(v21 - 32);
    if ( !v22 )
      return 0;
    if ( *(_BYTE *)v22 )
      return 0;
    if ( *(_QWORD *)(v22 + 24) != *(_QWORD *)(v21 + 80) )
      return 0;
    if ( *(_DWORD *)(v22 + 36) != *((_DWORD *)a1 + 2) )
      return 0;
    v23 = *(_QWORD *)(v21 + 32 * (*((unsigned int *)a1 + 4) - (unsigned __int64)(*(_DWORD *)(v21 + 4) & 0x7FFFFFF)));
    if ( !v23 )
      return 0;
    *(_QWORD *)a1[3] = v23;
    v24 = *((_QWORD *)v20 - 4);
    if ( *(_BYTE *)v24 == 17 )
    {
      v25 = *(_DWORD *)(v24 + 32);
      if ( v25 > 0x40 )
      {
        v56 = a3;
        v26 = sub_C444A0(v24 + 24);
        a3 = v56;
        if ( v26 == v25 - 1 )
          goto LABEL_38;
        return 0;
      }
      if ( *(_QWORD *)(v24 + 24) != 1 )
        return 0;
    }
    else
    {
      v34 = *(_QWORD *)(v24 + 8);
      v58 = a3;
      if ( (unsigned int)*(unsigned __int8 *)(v34 + 8) - 17 > 1 || *(_BYTE *)v24 > 0x15u )
        return 0;
      v35 = sub_AD7630(v24, 0, (__int64)a3);
      a3 = v58;
      if ( v35 && *v35 == 17 )
      {
        v36 = *((_DWORD *)v35 + 8);
        if ( v36 <= 0x40 )
        {
          v38 = *((_QWORD *)v35 + 3) == 1;
        }
        else
        {
          v37 = sub_C444A0((__int64)(v35 + 24));
          a3 = v58;
          v38 = v36 - 1 == v37;
        }
        if ( !v38 )
          return 0;
      }
      else
      {
        if ( *(_BYTE *)(v34 + 8) != 17 )
          return 0;
        v45 = *(_DWORD *)(v34 + 32);
        v46 = 0;
        v47 = 0;
        v60 = v45;
        while ( v60 != v46 )
        {
          v52 = a3;
          v48 = sub_AD69F0((unsigned __int8 *)v24, v46);
          if ( !v48 )
            return 0;
          a3 = v52;
          if ( *(_BYTE *)v48 != 13 )
          {
            if ( *(_BYTE *)v48 != 17 )
              return 0;
            v49 = *(_DWORD *)(v48 + 32);
            if ( v49 <= 0x40 )
            {
              v47 = *(_QWORD *)(v48 + 24) == 1;
            }
            else
            {
              v50 = sub_C444A0(v48 + 24);
              a3 = v52;
              v47 = v49 - 1 == v50;
            }
            if ( !v47 )
              return 0;
          }
          ++v46;
        }
        if ( !v47 )
          return 0;
      }
    }
LABEL_38:
    v27 = (__int64 *)a1[4];
    if ( v27 )
      *v27 = v24;
    if ( *a1 )
    {
      v57 = a3;
      v28 = sub_B53900((__int64)v20);
      v29 = *a1;
      a3 = v57;
      *(_DWORD *)v29 = v28;
      *(_BYTE *)(v29 + 4) = BYTE4(v28);
    }
    v30 = *((_QWORD *)a3 - 8);
    v31 = *(_QWORD *)(v30 + 16);
    if ( v31 && !*(_QWORD *)(v31 + 8) )
      return sub_10A3FD0((__int64)(a1 + 5), v30);
    return 0;
  }
  v55 = a3;
  result = sub_10A3FD0((__int64)(a1 + 5), v6);
  a3 = v55;
  if ( !(_BYTE)result )
  {
LABEL_4:
    v6 = *((_QWORD *)a3 - 4);
    goto LABEL_5;
  }
  return result;
}
