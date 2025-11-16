// Function: sub_10A8EC0
// Address: 0x10a8ec0
//
__int64 __fastcall sub_10A8EC0(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  _QWORD *v6; // rsi
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // r13
  unsigned int v11; // r12d
  int v12; // eax
  __int64 *v13; // rax
  _QWORD *v14; // rsi
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // r13
  unsigned int v19; // r12d
  int v20; // eax
  bool v21; // al
  __int64 *v22; // rax
  __int64 v23; // r12
  _BYTE *v24; // rax
  unsigned int v25; // r12d
  int v26; // eax
  __int64 v27; // r12
  _BYTE *v28; // rax
  unsigned int v29; // r12d
  int v30; // eax
  bool v31; // al
  int v32; // r14d
  bool v33; // r12
  unsigned int v34; // r15d
  __int64 v35; // rax
  unsigned int v36; // r12d
  int v37; // eax
  int v38; // r14d
  bool v39; // r12
  unsigned int v40; // r15d
  __int64 v41; // rax
  unsigned int v42; // r12d
  int v43; // eax
  unsigned __int8 *v44; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v45; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v46; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v47; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v48; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v49; // [rsp-40h] [rbp-40h]

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  if ( *(_BYTE *)v4 != 86 )
    goto LABEL_4;
  v14 = (*(_BYTE *)(v4 + 7) & 0x40) != 0
      ? *(_QWORD **)(v4 - 8)
      : (_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF));
  if ( !*v14 )
    goto LABEL_4;
  **a1 = *v14;
  v15 = (*(_BYTE *)(v4 + 7) & 0x40) != 0 ? *(_QWORD *)(v4 - 8) : v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
  v16 = *(_QWORD *)(v15 + 32);
  if ( !v16 )
    goto LABEL_4;
  *a1[1] = v16;
  if ( (*(_BYTE *)(v4 + 7) & 0x40) != 0 )
    v17 = *(_QWORD *)(v4 - 8);
  else
    v17 = v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
  v18 = *(_QWORD *)(v17 + 64);
  if ( *(_BYTE *)v18 == 17 )
  {
    v19 = *(_DWORD *)(v18 + 32);
    if ( v19 <= 0x40 )
    {
      v21 = *(_QWORD *)(v18 + 24) == 0;
    }
    else
    {
      v45 = a3;
      v20 = sub_C444A0(v18 + 24);
      a3 = v45;
      v21 = v19 == v20;
    }
  }
  else
  {
    v23 = *(_QWORD *)(v18 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v23 + 8) - 17 > 1 || *(_BYTE *)v18 > 0x15u )
      goto LABEL_4;
    v46 = a3;
    v24 = sub_AD7630(v18, 0, (__int64)a3);
    a3 = v46;
    if ( !v24 || *v24 != 17 )
    {
      if ( *(_BYTE *)(v23 + 8) == 17 )
      {
        v32 = *(_DWORD *)(v23 + 32);
        if ( v32 )
        {
          v33 = 0;
          v34 = 0;
          while ( 1 )
          {
            v48 = a3;
            v35 = sub_AD69F0((unsigned __int8 *)v18, v34);
            a3 = v48;
            if ( !v35 )
              break;
            if ( *(_BYTE *)v35 != 13 )
            {
              if ( *(_BYTE *)v35 != 17 )
                break;
              v36 = *(_DWORD *)(v35 + 32);
              if ( v36 <= 0x40 )
              {
                v33 = *(_QWORD *)(v35 + 24) == 0;
              }
              else
              {
                v37 = sub_C444A0(v35 + 24);
                a3 = v48;
                v33 = v36 == v37;
              }
              if ( !v33 )
                break;
            }
            if ( v32 == ++v34 )
            {
              if ( v33 )
                goto LABEL_34;
              goto LABEL_4;
            }
          }
        }
      }
      goto LABEL_4;
    }
    v25 = *((_DWORD *)v24 + 8);
    if ( v25 <= 0x40 )
    {
      v21 = *((_QWORD *)v24 + 3) == 0;
    }
    else
    {
      v26 = sub_C444A0((__int64)(v24 + 24));
      a3 = v46;
      v21 = v25 == v26;
    }
  }
  if ( !v21 )
  {
LABEL_4:
    v5 = *((_QWORD *)a3 - 4);
    goto LABEL_5;
  }
LABEL_34:
  v22 = a1[2];
  if ( v22 )
    *v22 = v18;
  v5 = *((_QWORD *)a3 - 4);
  if ( v5 )
    goto LABEL_21;
LABEL_5:
  if ( *(_BYTE *)v5 != 86 )
    return 0;
  v6 = (*(_BYTE *)(v5 + 7) & 0x40) != 0
     ? *(_QWORD **)(v5 - 8)
     : (_QWORD *)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
  if ( !*v6 )
    return 0;
  **a1 = *v6;
  v7 = (*(_BYTE *)(v5 + 7) & 0x40) != 0 ? *(_QWORD *)(v5 - 8) : v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
  v8 = *(_QWORD *)(v7 + 32);
  if ( !v8 )
    return 0;
  *a1[1] = v8;
  if ( (*(_BYTE *)(v5 + 7) & 0x40) != 0 )
    v9 = *(_QWORD *)(v5 - 8);
  else
    v9 = v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
  v10 = *(_QWORD *)(v9 + 64);
  if ( *(_BYTE *)v10 == 17 )
  {
    v11 = *(_DWORD *)(v10 + 32);
    if ( v11 > 0x40 )
    {
      v44 = a3;
      v12 = sub_C444A0(v10 + 24);
      a3 = v44;
      if ( v11 == v12 )
        goto LABEL_18;
      return 0;
    }
    if ( *(_QWORD *)(v10 + 24) )
      return 0;
  }
  else
  {
    v27 = *(_QWORD *)(v10 + 8);
    v47 = a3;
    if ( (unsigned int)*(unsigned __int8 *)(v27 + 8) - 17 > 1 || *(_BYTE *)v10 > 0x15u )
      return 0;
    v28 = sub_AD7630(v10, 0, (__int64)a3);
    a3 = v47;
    if ( !v28 || *v28 != 17 )
    {
      if ( *(_BYTE *)(v27 + 8) == 17 )
      {
        v38 = *(_DWORD *)(v27 + 32);
        if ( v38 )
        {
          v39 = 0;
          v40 = 0;
          while ( 1 )
          {
            v49 = a3;
            v41 = sub_AD69F0((unsigned __int8 *)v10, v40);
            if ( !v41 )
              break;
            a3 = v49;
            if ( *(_BYTE *)v41 != 13 )
            {
              if ( *(_BYTE *)v41 != 17 )
                break;
              v42 = *(_DWORD *)(v41 + 32);
              if ( v42 <= 0x40 )
              {
                v39 = *(_QWORD *)(v41 + 24) == 0;
              }
              else
              {
                v43 = sub_C444A0(v41 + 24);
                a3 = v49;
                v39 = v42 == v43;
              }
              if ( !v39 )
                break;
            }
            if ( v38 == ++v40 )
            {
              if ( !v39 )
                return 0;
              goto LABEL_18;
            }
          }
        }
      }
      return 0;
    }
    v29 = *((_DWORD *)v28 + 8);
    if ( v29 <= 0x40 )
    {
      v31 = *((_QWORD *)v28 + 3) == 0;
    }
    else
    {
      v30 = sub_C444A0((__int64)(v28 + 24));
      a3 = v47;
      v31 = v29 == v30;
    }
    if ( !v31 )
      return 0;
  }
LABEL_18:
  v13 = a1[2];
  if ( v13 )
    *v13 = v10;
  v5 = *((_QWORD *)a3 - 8);
  if ( !v5 )
    return 0;
LABEL_21:
  *a1[3] = v5;
  return 1;
}
