// Function: sub_2779B30
// Address: 0x2779b30
//
__int64 __fastcall sub_2779B30(__int64 a1, __int64 *a2, _QWORD *a3, _QWORD *a4, _DWORD *a5)
{
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  _BYTE *v13; // r12
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rcx
  unsigned int v18; // eax
  unsigned int v19; // eax
  _DWORD *v20; // r8
  __int64 v21; // r9
  unsigned int v22; // ebx
  __int64 v23; // rdi
  int v24; // eax
  bool v25; // al
  __int64 v26; // rbx
  unsigned int v27; // r14d
  int v28; // eax
  bool v29; // al
  __int64 v30; // rax
  __int64 v31; // rbx
  _BYTE *v32; // rax
  unsigned __int8 *v33; // r9
  int v34; // eax
  bool v35; // r14
  unsigned int v36; // ebx
  __int64 v37; // rax
  unsigned int v38; // r14d
  int v39; // eax
  __int64 v40; // r14
  _BYTE *v41; // rax
  unsigned int v42; // ebx
  int v43; // eax
  bool v44; // r14
  __int64 v45; // rsi
  __int64 v46; // rax
  unsigned int v47; // r14d
  int v48; // eax
  int v49; // [rsp-54h] [rbp-54h]
  int v50; // [rsp-54h] [rbp-54h]
  _DWORD *v51; // [rsp-50h] [rbp-50h]
  _DWORD *v52; // [rsp-50h] [rbp-50h]
  _DWORD *v53; // [rsp-50h] [rbp-50h]
  _DWORD *v54; // [rsp-48h] [rbp-48h]
  _DWORD *v55; // [rsp-48h] [rbp-48h]
  _QWORD *v56; // [rsp-48h] [rbp-48h]
  _QWORD *v57; // [rsp-48h] [rbp-48h]
  _DWORD *v58; // [rsp-48h] [rbp-48h]
  _QWORD *v59; // [rsp-48h] [rbp-48h]
  _DWORD *v60; // [rsp-40h] [rbp-40h]
  _QWORD *v61; // [rsp-40h] [rbp-40h]
  _QWORD *v62; // [rsp-40h] [rbp-40h]
  __int64 v63; // [rsp-40h] [rbp-40h]
  _DWORD *v64; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v65; // [rsp-40h] [rbp-40h]
  _QWORD *v66; // [rsp-40h] [rbp-40h]

  if ( *(_BYTE *)a1 != 86 )
    return 0;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) == 0 )
  {
    v8 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
    if ( v8 )
      goto LABEL_5;
    return 0;
  }
  v8 = **(_QWORD **)(a1 - 8);
  if ( !v8 )
    return 0;
LABEL_5:
  *a2 = v8;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v9 = *(_QWORD *)(a1 - 8);
  else
    v9 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  v10 = *(_QWORD *)(v9 + 32);
  if ( !v10 )
    return 0;
  *a3 = v10;
  v11 = (*(_BYTE *)(a1 + 7) & 0x40) != 0 ? *(_QWORD *)(a1 - 8) : a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  v12 = *(_QWORD *)(v11 + 64);
  if ( !v12 )
    return 0;
  *a4 = v12;
  v13 = (_BYTE *)*a2;
  if ( *(_BYTE *)*a2 != 59 )
    goto LABEL_12;
  v21 = *((_QWORD *)v13 - 8);
  if ( *(_BYTE *)v21 == 17 )
  {
    v22 = *(_DWORD *)(v21 + 32);
    if ( !v22 )
      goto LABEL_61;
    if ( v22 > 0x40 )
    {
      v54 = a5;
      v23 = v21 + 24;
      v61 = a3;
LABEL_30:
      v24 = sub_C445E0(v23);
      a3 = v61;
      a5 = v54;
      v25 = v22 == v24;
      goto LABEL_31;
    }
    v25 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v22) == *(_QWORD *)(v21 + 24);
    goto LABEL_31;
  }
  v31 = *(_QWORD *)(v21 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v31 + 8) - 17 > 1 || *(_BYTE *)v21 > 0x15u )
    goto LABEL_32;
  v51 = a5;
  v56 = a3;
  v63 = *((_QWORD *)v13 - 8);
  v32 = sub_AD7630(v63, 0, (__int64)a3);
  v33 = (unsigned __int8 *)v63;
  a3 = v56;
  a5 = v51;
  if ( !v32 || *v32 != 17 )
  {
    if ( *(_BYTE *)(v31 + 8) == 17 )
    {
      v34 = *(_DWORD *)(v31 + 32);
      v35 = 0;
      v36 = 0;
      v49 = v34;
      if ( v34 )
      {
        while ( 1 )
        {
          v52 = a5;
          v57 = a3;
          v65 = v33;
          v37 = sub_AD69F0(v33, v36);
          v33 = v65;
          a3 = v57;
          a5 = v52;
          if ( !v37 )
            break;
          if ( *(_BYTE *)v37 != 13 )
          {
            if ( *(_BYTE *)v37 != 17 )
              goto LABEL_32;
            v38 = *(_DWORD *)(v37 + 32);
            if ( v38 )
            {
              if ( v38 <= 0x40 )
              {
                v35 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v38) == *(_QWORD *)(v37 + 24);
              }
              else
              {
                v39 = sub_C445E0(v37 + 24);
                v33 = v65;
                a3 = v57;
                a5 = v52;
                v35 = v38 == v39;
              }
              if ( !v35 )
                goto LABEL_32;
            }
            else
            {
              v35 = 1;
            }
          }
          if ( v49 == ++v36 )
          {
            if ( !v35 )
              goto LABEL_32;
            goto LABEL_61;
          }
        }
      }
    }
    goto LABEL_32;
  }
  v22 = *((_DWORD *)v32 + 8);
  if ( v22 )
  {
    if ( v22 > 0x40 )
    {
      v54 = v51;
      v23 = (__int64)(v32 + 24);
      v61 = a3;
      goto LABEL_30;
    }
    v25 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v22) == *((_QWORD *)v32 + 3);
LABEL_31:
    if ( !v25 )
    {
LABEL_32:
      v26 = *((_QWORD *)v13 - 4);
      goto LABEL_33;
    }
  }
LABEL_61:
  v26 = *((_QWORD *)v13 - 4);
  if ( v26 )
  {
LABEL_39:
    *a2 = v26;
    v30 = *a3;
    *a3 = *a4;
    *a4 = v30;
    goto LABEL_12;
  }
LABEL_33:
  if ( *(_BYTE *)v26 == 17 )
  {
    v27 = *(_DWORD *)(v26 + 32);
    if ( !v27
      || (v27 <= 0x40
        ? (v29 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v27) == *(_QWORD *)(v26 + 24))
        : (v55 = a5, v62 = a3, v28 = sub_C445E0(v26 + 24), a3 = v62, a5 = v55, v29 = v27 == v28),
          v29) )
    {
LABEL_38:
      v26 = *((_QWORD *)v13 - 8);
      if ( !v26 )
        goto LABEL_12;
      goto LABEL_39;
    }
  }
  else
  {
    v40 = *(_QWORD *)(v26 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v40 + 8) - 17 <= 1 && *(_BYTE *)v26 <= 0x15u )
    {
      v58 = a5;
      v66 = a3;
      v41 = sub_AD7630(v26, 0, (__int64)a3);
      a3 = v66;
      a5 = v58;
      if ( v41 && *v41 == 17 )
      {
        v42 = *((_DWORD *)v41 + 8);
        if ( !v42 )
          goto LABEL_38;
        if ( v42 <= 0x40 )
        {
          if ( *((_QWORD *)v41 + 3) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v42) )
            goto LABEL_38;
        }
        else
        {
          v43 = sub_C445E0((__int64)(v41 + 24));
          a3 = v66;
          a5 = v58;
          if ( v42 == v43 )
            goto LABEL_38;
        }
      }
      else if ( *(_BYTE *)(v40 + 8) == 17 )
      {
        v50 = *(_DWORD *)(v40 + 32);
        if ( v50 )
        {
          v44 = 0;
          v45 = 0;
          while ( 1 )
          {
            v53 = a5;
            v59 = a3;
            v46 = sub_AD69F0((unsigned __int8 *)v26, v45);
            a3 = v59;
            a5 = v53;
            if ( !v46 )
              break;
            if ( *(_BYTE *)v46 != 13 )
            {
              if ( *(_BYTE *)v46 != 17 )
                break;
              v47 = *(_DWORD *)(v46 + 32);
              if ( v47 )
              {
                if ( v47 <= 0x40 )
                {
                  v44 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v47) == *(_QWORD *)(v46 + 24);
                }
                else
                {
                  v48 = sub_C445E0(v46 + 24);
                  a3 = v59;
                  a5 = v53;
                  v44 = v47 == v48;
                }
                if ( !v44 )
                  break;
              }
              else
              {
                v44 = 1;
              }
            }
            v45 = (unsigned int)(v45 + 1);
            if ( v50 == (_DWORD)v45 )
            {
              if ( v44 )
                goto LABEL_38;
              break;
            }
          }
        }
      }
    }
  }
LABEL_12:
  *a5 = 0;
  v14 = *a2;
  if ( *(_BYTE *)*a2 != 82 )
    return 1;
  v15 = *a3;
  v16 = *(_QWORD *)(v14 - 64);
  v17 = *a4;
  if ( v15 == v16 && v17 == *(_QWORD *)(v14 - 32) )
  {
    v64 = a5;
    v19 = sub_B53900(v14);
    v20 = v64;
  }
  else
  {
    if ( v17 != v16 || v15 != *(_QWORD *)(v14 - 32) )
      return 1;
    v60 = a5;
    v18 = sub_B53900(v14);
    v19 = sub_B52F50(v18);
    v20 = v60;
  }
  if ( v19 > 0x27 )
  {
    if ( v19 - 40 <= 1 )
      *v20 = 1;
    return 1;
  }
  if ( v19 > 0x25 )
  {
    *v20 = 3;
    return 1;
  }
  if ( v19 <= 0x23 )
  {
    if ( v19 > 0x21 )
      *v20 = 4;
    return 1;
  }
  *v20 = 2;
  return 1;
}
