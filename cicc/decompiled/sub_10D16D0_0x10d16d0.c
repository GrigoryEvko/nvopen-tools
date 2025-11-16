// Function: sub_10D16D0
// Address: 0x10d16d0
//
bool __fastcall sub_10D16D0(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  bool result; // al
  __int64 v5; // r13
  __int64 v6; // rax
  unsigned __int8 *v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdi
  bool v10; // al
  __int64 v11; // r14
  _BYTE *v12; // rdi
  __int64 v13; // r13
  bool v14; // al
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // r8
  __int64 v18; // rcx
  __int64 v19; // r14
  _BYTE *v20; // rdi
  __int64 v21; // r13
  __int64 v22; // rdi
  unsigned __int8 *v23; // rdx
  __int64 v24; // r14
  _BYTE *v25; // rdi
  __int64 v26; // r13
  unsigned __int8 *v27; // rbx
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // rdx
  __int64 v33; // r13
  _BYTE *v34; // rdi
  __int64 v35; // rbx
  _QWORD *v36; // r13
  __int64 v37; // rcx
  unsigned __int8 *v38; // r13
  __int64 v39; // rcx
  unsigned __int8 *v40; // r13
  __int64 v41; // rcx
  unsigned __int8 *v42; // rbx
  unsigned __int8 *v43; // [rsp-30h] [rbp-30h]
  __int64 v44; // [rsp-30h] [rbp-30h]
  unsigned __int8 *v45; // [rsp-30h] [rbp-30h]

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = *((_QWORD *)a3 - 8);
  v6 = *(_QWORD *)(v5 + 16);
  if ( !v6 || *(_QWORD *)(v6 + 8) || *(_BYTE *)v5 <= 0x1Cu )
    goto LABEL_4;
  v9 = *(_QWORD *)(v5 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
    v9 = **(_QWORD **)(v9 + 16);
  v43 = a3;
  v10 = sub_BCAC40(v9, 1);
  a3 = v43;
  if ( !v10 )
    goto LABEL_4;
  if ( *(_BYTE *)v5 == 57 )
  {
    if ( (*(_BYTE *)(v5 + 7) & 0x40) != 0 )
      v36 = *(_QWORD **)(v5 - 8);
    else
      v36 = (_QWORD *)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
    if ( *v36 )
    {
      v37 = v36[4];
      **a1 = *v36;
      if ( v37 )
      {
        *a1[1] = v37;
        goto LABEL_18;
      }
    }
LABEL_4:
    v7 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
    v8 = *((_QWORD *)v7 + 2);
    if ( !v8 || *(_QWORD *)(v8 + 8) )
      return 0;
    goto LABEL_32;
  }
  if ( *(_BYTE *)v5 != 86 )
    goto LABEL_4;
  v11 = *(_QWORD *)(v5 - 96);
  if ( *(_QWORD *)(v11 + 8) != *(_QWORD *)(v5 + 8) )
    goto LABEL_4;
  v12 = *(_BYTE **)(v5 - 32);
  if ( *v12 > 0x15u )
    goto LABEL_4;
  v13 = *(_QWORD *)(v5 - 64);
  v14 = sub_AC30F0((__int64)v12);
  a3 = v43;
  if ( !v14 )
    goto LABEL_4;
  **a1 = v11;
  if ( !v13 )
    goto LABEL_4;
  *a1[1] = v13;
LABEL_18:
  v7 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
  v15 = *((_QWORD *)v7 + 2);
  if ( !v15 || *(_QWORD *)(v15 + 8) )
    return 0;
  if ( *v7 > 0x1Cu )
  {
    v16 = *((_QWORD *)v7 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 <= 1 )
      v16 = **(_QWORD **)(v16 + 16);
    v44 = (__int64)a3;
    result = sub_BCAC40(v16, 1);
    a3 = (unsigned __int8 *)v44;
    if ( result )
    {
      v18 = *v7;
      if ( (_BYTE)v18 == 58 )
      {
        if ( (v7[7] & 0x40) != 0 )
          v40 = (unsigned __int8 *)*((_QWORD *)v7 - 1);
        else
          v40 = &v7[-32 * (*((_DWORD *)v7 + 1) & 0x7FFFFFF)];
        if ( *(_QWORD *)v40 )
        {
          v41 = *((_QWORD *)v40 + 4);
          *a1[2] = *(_QWORD *)v40;
          if ( v41 )
            goto LABEL_70;
        }
      }
      else if ( (_BYTE)v18 == 86 )
      {
        v19 = *((_QWORD *)v7 - 12);
        if ( *(_QWORD *)(v19 + 8) == *((_QWORD *)v7 + 1) )
        {
          v20 = (_BYTE *)*((_QWORD *)v7 - 8);
          if ( *v20 <= 0x15u )
          {
            v21 = *((_QWORD *)v7 - 4);
            result = sub_AD7A80(v20, 1, v44, v18, v17);
            a3 = (unsigned __int8 *)v44;
            if ( result )
            {
              *a1[2] = v19;
              if ( v21 )
              {
                *a1[3] = v21;
                return result;
              }
            }
          }
        }
      }
    }
    goto LABEL_4;
  }
LABEL_32:
  if ( *v7 <= 0x1Cu )
    return 0;
  v22 = *((_QWORD *)v7 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v22 + 8) - 17 <= 1 )
    v22 = **(_QWORD **)(v22 + 16);
  v45 = a3;
  if ( !sub_BCAC40(v22, 1) )
    return 0;
  v23 = v45;
  if ( *v7 == 57 )
  {
    if ( (v7[7] & 0x40) != 0 )
      v38 = (unsigned __int8 *)*((_QWORD *)v7 - 1);
    else
      v38 = &v7[-32 * (*((_DWORD *)v7 + 1) & 0x7FFFFFF)];
    if ( !*(_QWORD *)v38 )
      return 0;
    v39 = *((_QWORD *)v38 + 4);
    **a1 = *(_QWORD *)v38;
    if ( !v39 )
      return 0;
    *a1[1] = v39;
  }
  else
  {
    if ( *v7 != 86 )
      return 0;
    v24 = *((_QWORD *)v7 - 12);
    if ( *(_QWORD *)(v24 + 8) != *((_QWORD *)v7 + 1) )
      return 0;
    v25 = (_BYTE *)*((_QWORD *)v7 - 4);
    if ( *v25 > 0x15u )
      return 0;
    v26 = *((_QWORD *)v7 - 8);
    if ( !sub_AC30F0((__int64)v25) )
      return 0;
    **a1 = v24;
    if ( !v26 )
      return 0;
    v23 = v45;
    *a1[1] = v26;
  }
  v27 = (unsigned __int8 *)*((_QWORD *)v23 - 8);
  v28 = *((_QWORD *)v27 + 2);
  if ( !v28 || *(_QWORD *)(v28 + 8) || *v27 <= 0x1Cu )
    return 0;
  v29 = *((_QWORD *)v27 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v29 + 8) - 17 <= 1 )
    v29 = **(_QWORD **)(v29 + 16);
  result = sub_BCAC40(v29, 1);
  if ( !result )
    return 0;
  v32 = *v27;
  if ( (_BYTE)v32 != 58 )
  {
    if ( (_BYTE)v32 == 86 )
    {
      v33 = *((_QWORD *)v27 - 12);
      if ( *(_QWORD *)(v33 + 8) == *((_QWORD *)v27 + 1) )
      {
        v34 = (_BYTE *)*((_QWORD *)v27 - 8);
        if ( *v34 <= 0x15u )
        {
          v35 = *((_QWORD *)v27 - 4);
          result = sub_AD7A80(v34, 1, v32, v30, v31);
          if ( result )
          {
            *a1[2] = v33;
            if ( v35 )
            {
              *a1[3] = v35;
              return result;
            }
          }
        }
      }
    }
    return 0;
  }
  if ( (v27[7] & 0x40) != 0 )
    v42 = (unsigned __int8 *)*((_QWORD *)v27 - 1);
  else
    v42 = &v27[-32 * (*((_DWORD *)v27 + 1) & 0x7FFFFFF)];
  if ( !*(_QWORD *)v42 )
    return 0;
  v41 = *((_QWORD *)v42 + 4);
  *a1[2] = *(_QWORD *)v42;
  if ( !v41 )
    return 0;
LABEL_70:
  *a1[3] = v41;
  return result;
}
