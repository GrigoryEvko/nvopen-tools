// Function: sub_23D36F0
// Address: 0x23d36f0
//
__int64 __fastcall sub_23D36F0(__int64 a1, int a2, unsigned __int8 *a3)
{
  _BYTE *v4; // r15
  _BYTE *v5; // r12
  _BYTE *v6; // rcx
  _BYTE *v7; // r13
  __int64 v8; // rcx
  unsigned int v9; // r14d
  int v10; // eax
  __int64 v11; // rcx
  __int64 v12; // r12
  unsigned int v13; // r13d
  _QWORD *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r12
  unsigned int v17; // r13d
  _QWORD *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r8
  unsigned int v21; // r13d
  int v22; // eax
  bool v23; // al
  __int64 v24; // rcx
  __int64 v25; // r13
  _BYTE *v26; // rax
  unsigned int v27; // r13d
  int v28; // eax
  __int64 v29; // r14
  _BYTE *v30; // rax
  unsigned int v31; // r14d
  int v32; // eax
  bool v33; // al
  bool v34; // r14
  unsigned int v35; // r13d
  __int64 v36; // rax
  unsigned int v37; // r14d
  int v38; // eax
  bool v39; // r8
  unsigned int v40; // r14d
  __int64 v41; // rax
  int v42; // eax
  unsigned __int8 *v43; // [rsp-58h] [rbp-58h]
  unsigned __int8 *v44; // [rsp-58h] [rbp-58h]
  unsigned __int8 *v45; // [rsp-50h] [rbp-50h]
  unsigned __int8 *v46; // [rsp-50h] [rbp-50h]
  _BYTE *v47; // [rsp-50h] [rbp-50h]
  __int64 v48; // [rsp-50h] [rbp-50h]
  bool v49; // [rsp-50h] [rbp-50h]
  int v50; // [rsp-50h] [rbp-50h]
  unsigned __int8 *v51; // [rsp-48h] [rbp-48h]
  _BYTE *v52; // [rsp-48h] [rbp-48h]
  _BYTE *v53; // [rsp-48h] [rbp-48h]
  __int64 v54; // [rsp-48h] [rbp-48h]
  __int64 v55; // [rsp-48h] [rbp-48h]
  unsigned __int8 *v56; // [rsp-48h] [rbp-48h]
  __int64 v57; // [rsp-48h] [rbp-48h]
  _BYTE *v58; // [rsp-48h] [rbp-48h]
  __int64 v59; // [rsp-48h] [rbp-48h]
  __int64 v60; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v61; // [rsp-40h] [rbp-40h]
  __int64 v62; // [rsp-40h] [rbp-40h]
  __int64 v63; // [rsp-40h] [rbp-40h]
  _BYTE *v64; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v65; // [rsp-40h] [rbp-40h]
  __int64 v66; // [rsp-40h] [rbp-40h]
  int v67; // [rsp-40h] [rbp-40h]
  int v68; // [rsp-40h] [rbp-40h]

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v4 != 46 )
    return 0;
  v5 = (_BYTE *)*((_QWORD *)v4 - 8);
  if ( *v5 != 57 )
    return 0;
  v6 = (_BYTE *)*((_QWORD *)v5 - 8);
  if ( *v6 != 44 )
    goto LABEL_7;
  v20 = *((_QWORD *)v6 - 8);
  if ( *(_BYTE *)v20 == 17 )
  {
    v21 = *(_DWORD *)(v20 + 32);
    if ( v21 <= 0x40 )
    {
      v23 = *(_QWORD *)(v20 + 24) == 0;
    }
    else
    {
      v45 = a3;
      v52 = (_BYTE *)*((_QWORD *)v5 - 8);
      v62 = *((_QWORD *)v6 - 8);
      v22 = sub_C444A0(v20 + 24);
      v20 = v62;
      v6 = v52;
      a3 = v45;
      v23 = v21 == v22;
    }
  }
  else
  {
    v25 = *(_QWORD *)(v20 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v25 + 8) - 17 > 1 || *(_BYTE *)v20 > 0x15u )
      goto LABEL_7;
    v46 = a3;
    v53 = (_BYTE *)*((_QWORD *)v5 - 8);
    v63 = *((_QWORD *)v6 - 8);
    v26 = sub_AD7630(v63, 0, (__int64)a3);
    v20 = v63;
    v6 = v53;
    a3 = v46;
    if ( !v26 || *v26 != 17 )
    {
      if ( *(_BYTE *)(v25 + 8) == 17 )
      {
        v67 = *(_DWORD *)(v25 + 32);
        if ( v67 )
        {
          v34 = 0;
          v35 = 0;
          while ( 1 )
          {
            v43 = a3;
            v47 = v6;
            v57 = v20;
            v36 = sub_AD69F0((unsigned __int8 *)v20, v35);
            v20 = v57;
            v6 = v47;
            a3 = v43;
            if ( !v36 )
              break;
            if ( *(_BYTE *)v36 != 13 )
            {
              if ( *(_BYTE *)v36 != 17 )
                break;
              v37 = *(_DWORD *)(v36 + 32);
              if ( v37 <= 0x40 )
              {
                v34 = *(_QWORD *)(v36 + 24) == 0;
              }
              else
              {
                v48 = v57;
                v58 = v6;
                v38 = sub_C444A0(v36 + 24);
                v6 = v58;
                v20 = v48;
                a3 = v43;
                v34 = v37 == v38;
              }
              if ( !v34 )
                break;
            }
            if ( v67 == ++v35 )
            {
              if ( v34 )
                goto LABEL_31;
              goto LABEL_7;
            }
          }
        }
      }
      goto LABEL_7;
    }
    v27 = *((_DWORD *)v26 + 8);
    if ( v27 <= 0x40 )
    {
      v23 = *((_QWORD *)v26 + 3) == 0;
    }
    else
    {
      v54 = v63;
      v64 = v6;
      v28 = sub_C444A0((__int64)(v26 + 24));
      v6 = v64;
      v20 = v54;
      a3 = v46;
      v23 = v27 == v28;
    }
  }
  if ( !v23 )
  {
LABEL_7:
    v7 = (_BYTE *)*((_QWORD *)v5 - 4);
    goto LABEL_8;
  }
LABEL_31:
  if ( *(_QWORD *)a1 )
    **(_QWORD **)a1 = v20;
  v24 = *((_QWORD *)v6 - 4);
  if ( !v24 )
    goto LABEL_7;
  **(_QWORD **)(a1 + 8) = v24;
  v7 = (_BYTE *)*((_QWORD *)v5 - 4);
  if ( v7 != **(_BYTE ***)(a1 + 16) )
  {
LABEL_8:
    if ( *v7 != 44 )
      return 0;
    v8 = *((_QWORD *)v7 - 8);
    if ( *(_BYTE *)v8 == 17 )
    {
      v9 = *(_DWORD *)(v8 + 32);
      if ( v9 > 0x40 )
      {
        v51 = a3;
        v60 = *((_QWORD *)v7 - 8);
        v10 = sub_C444A0(v8 + 24);
        v8 = v60;
        a3 = v51;
        if ( v9 != v10 )
          return 0;
        goto LABEL_12;
      }
      v33 = *(_QWORD *)(v8 + 24) == 0;
    }
    else
    {
      v29 = *(_QWORD *)(v8 + 8);
      v65 = a3;
      if ( (unsigned int)*(unsigned __int8 *)(v29 + 8) - 17 > 1 || *(_BYTE *)v8 > 0x15u )
        return 0;
      v55 = *((_QWORD *)v7 - 8);
      v30 = sub_AD7630(v8, 0, (__int64)a3);
      v8 = v55;
      a3 = v65;
      if ( !v30 || *v30 != 17 )
      {
        if ( *(_BYTE *)(v29 + 8) == 17 )
        {
          v68 = *(_DWORD *)(v29 + 32);
          if ( v68 )
          {
            v39 = 0;
            v40 = 0;
            while ( 1 )
            {
              v44 = a3;
              v49 = v39;
              v59 = v8;
              v41 = sub_AD69F0((unsigned __int8 *)v8, v40);
              if ( !v41 )
                break;
              v8 = v59;
              v39 = v49;
              a3 = v44;
              if ( *(_BYTE *)v41 != 13 )
              {
                if ( *(_BYTE *)v41 != 17 )
                  break;
                if ( *(_DWORD *)(v41 + 32) <= 0x40u )
                {
                  v39 = *(_QWORD *)(v41 + 24) == 0;
                }
                else
                {
                  v50 = *(_DWORD *)(v41 + 32);
                  v42 = sub_C444A0(v41 + 24);
                  a3 = v44;
                  v8 = v59;
                  v39 = v50 == v42;
                }
                if ( !v39 )
                  break;
              }
              if ( v68 == ++v40 )
              {
                if ( v39 )
                  goto LABEL_12;
                return 0;
              }
            }
          }
        }
        return 0;
      }
      v31 = *((_DWORD *)v30 + 8);
      if ( v31 <= 0x40 )
      {
        v33 = *((_QWORD *)v30 + 3) == 0;
      }
      else
      {
        v56 = v65;
        v66 = v8;
        v32 = sub_C444A0((__int64)(v30 + 24));
        v8 = v66;
        a3 = v56;
        v33 = v31 == v32;
      }
    }
    if ( !v33 )
      return 0;
LABEL_12:
    if ( *(_QWORD *)a1 )
      **(_QWORD **)a1 = v8;
    v11 = *((_QWORD *)v7 - 4);
    if ( !v11 )
      return 0;
    **(_QWORD **)(a1 + 8) = v11;
    if ( *((_QWORD *)v5 - 8) != **(_QWORD **)(a1 + 16) )
      return 0;
  }
  v12 = *((_QWORD *)v4 - 4);
  if ( *(_BYTE *)v12 != 17 )
    return 0;
  v13 = *(_DWORD *)(v12 + 32);
  if ( v13 > 0x40 )
  {
    v61 = a3;
    if ( v13 - (unsigned int)sub_C444A0(v12 + 24) > 0x40 )
      return 0;
    a3 = v61;
    v14 = *(_QWORD **)(a1 + 24);
    v15 = **(_QWORD **)(v12 + 24);
  }
  else
  {
    v14 = *(_QWORD **)(a1 + 24);
    v15 = *(_QWORD *)(v12 + 24);
  }
  *v14 = v15;
  v16 = *((_QWORD *)a3 - 4);
  if ( *(_BYTE *)v16 != 17 )
    return 0;
  v17 = *(_DWORD *)(v16 + 32);
  if ( v17 > 0x40 )
  {
    if ( v17 - (unsigned int)sub_C444A0(v16 + 24) <= 0x40 )
    {
      v18 = *(_QWORD **)(a1 + 32);
      v19 = **(_QWORD **)(v16 + 24);
      goto LABEL_22;
    }
    return 0;
  }
  v18 = *(_QWORD **)(a1 + 32);
  v19 = *(_QWORD *)(v16 + 24);
LABEL_22:
  *v18 = v19;
  return 1;
}
