// Function: sub_20D8130
// Address: 0x20d8130
//
__int64 __fastcall sub_20D8130(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rdi
  __int64 (*v3)(); // rax
  __int64 v5; // r8
  int v6; // r9d
  __int64 v7; // rdx
  __int64 v8; // r14
  unsigned __int8 v9; // bl
  _QWORD *v10; // r13
  __int64 *v11; // rsi
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // rbx
  __int64 v18; // rax
  _QWORD *v19; // rax
  _QWORD *v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 *v23; // r12
  __int64 *v24; // rbx
  __int64 *v25; // r15
  __int64 i; // r13
  __int64 v27; // r14
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdx
  _QWORD *v31; // rax
  __int64 v32; // rax
  __int64 v33; // rsi
  bool v34; // al
  unsigned __int64 v35; // r12
  __int64 *v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // [rsp+10h] [rbp-250h]
  __int64 *v41; // [rsp+28h] [rbp-238h]
  _QWORD *v42; // [rsp+40h] [rbp-220h]
  bool v44; // [rsp+56h] [rbp-20Ah]
  unsigned __int8 v45; // [rsp+57h] [rbp-209h]
  __int64 v47; // [rsp+60h] [rbp-200h] BYREF
  _QWORD *v48; // [rsp+68h] [rbp-1F8h] BYREF
  __int64 *v49; // [rsp+70h] [rbp-1F0h] BYREF
  __int64 v50; // [rsp+78h] [rbp-1E8h]
  _BYTE v51[48]; // [rsp+80h] [rbp-1E0h] BYREF
  __int64 *v52; // [rsp+B0h] [rbp-1B0h] BYREF
  __int64 v53; // [rsp+B8h] [rbp-1A8h]
  _BYTE v54[48]; // [rsp+C0h] [rbp-1A0h] BYREF
  __int64 *v55; // [rsp+F0h] [rbp-170h] BYREF
  __int64 v56; // [rsp+F8h] [rbp-168h]
  _BYTE v57[48]; // [rsp+100h] [rbp-160h] BYREF
  __int64 *v58; // [rsp+130h] [rbp-130h] BYREF
  unsigned __int64 v59; // [rsp+138h] [rbp-128h]
  __int64 v60; // [rsp+140h] [rbp-120h] BYREF
  _BYTE v61[32]; // [rsp+148h] [rbp-118h] BYREF
  unsigned __int64 v62; // [rsp+168h] [rbp-F8h]
  int v63; // [rsp+170h] [rbp-F0h]
  _BYTE *v64; // [rsp+180h] [rbp-E0h] BYREF
  __int64 v65; // [rsp+188h] [rbp-D8h]
  _BYTE v66[208]; // [rsp+190h] [rbp-D0h] BYREF

  v2 = *(_QWORD *)(a1 + 144);
  v64 = v66;
  v47 = 0;
  v48 = 0;
  v65 = 0x400000000LL;
  v3 = *(__int64 (**)())(*(_QWORD *)v2 + 264LL);
  v45 = 0;
  if ( v3 == sub_1D820E0 )
    return v45;
  v45 = ((__int64 (__fastcall *)(__int64, _QWORD *, __int64 *, _QWORD **, _BYTE **, __int64))v3)(
          v2,
          a2,
          &v47,
          &v48,
          &v64,
          1);
  if ( v45 || !v47 || !(_DWORD)v65 )
    goto LABEL_24;
  v7 = (__int64)v48;
  if ( !v48 )
  {
    v11 = (__int64 *)a2[12];
    v12 = (__int64 *)a2[11];
    if ( v12 == v11 )
      goto LABEL_24;
    while ( 1 )
    {
      v7 = *v12;
      if ( v47 != *v12 )
        break;
      if ( v11 == ++v12 )
        goto LABEL_24;
    }
    v48 = (_QWORD *)*v12;
    if ( !v7 )
      goto LABEL_24;
  }
  if ( (unsigned int)((__int64)(*(_QWORD *)(v7 + 72) - *(_QWORD *)(v7 + 64)) >> 3) > 1 )
    goto LABEL_24;
  v8 = a2[4];
  v42 = a2 + 3;
  if ( a2 + 3 == (_QWORD *)v8 )
    goto LABEL_24;
  do
  {
    while ( 1 )
    {
      v44 = (*(_QWORD *)(*(_QWORD *)(v8 + 16) + 8LL) & 0x400LL) != 0;
      if ( (*(_QWORD *)(*(_QWORD *)(v8 + 16) + 8LL) & 0x400LL) == 0 )
        goto LABEL_12;
      v49 = (__int64 *)v51;
      v50 = 0x600000000LL;
      v53 = 0x600000000LL;
      v56 = 0x600000000LL;
      v13 = *(_QWORD *)(v8 + 32);
      v52 = (__int64 *)v54;
      v14 = *(unsigned int *)(v13 + 8);
      v55 = (__int64 *)v57;
      v15 = *(_QWORD *)(a1 + 152);
      if ( (int)v14 < 0 )
        v16 = *(_QWORD *)(*(_QWORD *)(v15 + 24) + 16 * (v14 & 0x7FFFFFFF) + 8);
      else
        v16 = *(_QWORD *)(*(_QWORD *)(v15 + 272) + 8 * v14);
      if ( v16 )
      {
        while ( (*(_BYTE *)(v16 + 4) & 8) != 0 )
        {
          v16 = *(_QWORD *)(v16 + 32);
          if ( !v16 )
            goto LABEL_43;
        }
LABEL_34:
        v17 = *(_QWORD *)(v16 + 16);
        if ( (*(_BYTE *)(v16 + 3) & 0x10) == 0 )
        {
          if ( **(_WORD **)(v17 + 16) != 15 )
            goto LABEL_38;
          goto LABEL_55;
        }
        if ( !(unsigned __int8)sub_1E15D60(*(_QWORD *)(v16 + 16), v8, 1u) )
        {
          v19 = *(_QWORD **)(v17 + 24);
          if ( a2 == v19 )
          {
            if ( sub_20D6290(v8, v17) )
              goto LABEL_41;
            if ( v48 != *(_QWORD **)(v17 + 24) )
              goto LABEL_53;
          }
          else if ( v48 != v19 )
          {
            goto LABEL_53;
          }
          v22 = (unsigned int)v56;
          if ( (unsigned int)v56 >= HIDWORD(v56) )
          {
            sub_16CD150((__int64)&v55, v57, 0, 8, v5, v6);
            v22 = (unsigned int)v56;
          }
          v55[v22] = v17;
          LODWORD(v56) = v56 + 1;
        }
LABEL_53:
        if ( **(_WORD **)(v17 + 16) != 15 || (*(_BYTE *)(v16 + 3) & 0x10) != 0 )
          goto LABEL_38;
LABEL_55:
        v20 = *(_QWORD **)(v17 + 24);
        if ( a2 != v20 )
          goto LABEL_56;
        if ( !sub_20D6290(v8, v17) )
        {
          v20 = *(_QWORD **)(v17 + 24);
LABEL_56:
          if ( v48 == v20 )
          {
            v21 = (unsigned int)v53;
            if ( (unsigned int)v53 >= HIDWORD(v53) )
            {
              sub_16CD150((__int64)&v52, v54, 0, 8, v5, v6);
              v21 = (unsigned int)v53;
            }
            v52[v21] = v17;
            LODWORD(v53) = v53 + 1;
          }
          goto LABEL_38;
        }
        v39 = (unsigned int)v50;
        if ( (unsigned int)v50 >= HIDWORD(v50) )
        {
          sub_16CD150((__int64)&v49, v51, 0, 8, v5, v6);
          v39 = (unsigned int)v50;
        }
        v49[v39] = v17;
        LODWORD(v50) = v50 + 1;
LABEL_38:
        while ( 1 )
        {
          v16 = *(_QWORD *)(v16 + 32);
          if ( !v16 )
            break;
          if ( (*(_BYTE *)(v16 + 4) & 8) == 0 )
            goto LABEL_34;
        }
        v18 = (unsigned int)v50;
        if ( !(_DWORD)v50 || !(_DWORD)v53 )
          goto LABEL_41;
        v40 = v8;
        v58 = &v60;
        v23 = v52;
        v59 = 0x600000000LL;
        v41 = &v52[(unsigned int)v53];
        while ( 1 )
        {
          LODWORD(v5) = (_DWORD)v49;
          v24 = &v49[v18];
          v25 = v49;
          for ( i = *v23; v24 != v25; ++v25 )
          {
            v27 = *v25;
            if ( (unsigned __int8)sub_1E15D60(*v25, i, 1u) )
            {
              v28 = *(unsigned int *)(*(_QWORD *)(v27 + 32) + 8LL);
              v29 = *(_QWORD *)(a1 + 152);
              if ( (int)v28 < 0 )
                v30 = *(_QWORD *)(*(_QWORD *)(v29 + 24) + 16 * (v28 & 0x7FFFFFFF) + 8);
              else
                v30 = *(_QWORD *)(*(_QWORD *)(v29 + 272) + 8 * v28);
              if ( v30 )
              {
                if ( (*(_BYTE *)(v30 + 4) & 8) != 0 )
                {
                  while ( 1 )
                  {
                    v30 = *(_QWORD *)(v30 + 32);
                    if ( !v30 )
                      break;
                    if ( (*(_BYTE *)(v30 + 4) & 8) == 0 )
                      goto LABEL_78;
                  }
                }
                else
                {
LABEL_78:
                  v5 = *(_QWORD *)(v30 + 16);
                  if ( (*(_BYTE *)(v30 + 3) & 0x10) != 0 && i != v5 && v27 != v5 )
                  {
                    v31 = *(_QWORD **)(v5 + 24);
                    if ( a2 != v31 )
                    {
                      if ( v48 != v31 )
                        goto LABEL_84;
                      goto LABEL_93;
                    }
                    v33 = *(_QWORD *)(v30 + 16);
                    if ( sub_20D6290(v40, v33) )
                    {
                      v34 = sub_20D6290(v27, v33);
                    }
                    else
                    {
                      if ( v48 != *(_QWORD **)(v5 + 24) )
                        goto LABEL_84;
LABEL_93:
                      v34 = sub_20D6290(v5, i);
                    }
                    if ( !v34 )
                      goto LABEL_84;
                    continue;
                  }
LABEL_84:
                  while ( 1 )
                  {
                    v30 = *(_QWORD *)(v30 + 32);
                    if ( !v30 )
                      break;
                    if ( (*(_BYTE *)(v30 + 4) & 8) == 0 )
                      goto LABEL_78;
                  }
                }
              }
              v32 = (unsigned int)v59;
              if ( (unsigned int)v59 >= HIDWORD(v59) )
              {
                sub_16CD150((__int64)&v58, &v60, 0, 8, v5, v6);
                v32 = (unsigned int)v59;
              }
              v58[v32] = i;
              LODWORD(v59) = v59 + 1;
            }
          }
          if ( v41 == ++v23 )
            break;
          v18 = (unsigned int)v50;
        }
        v8 = v40;
        if ( (_DWORD)v59 )
        {
          v35 = 0;
          do
          {
            v36 = v55;
            v37 = v35;
            if ( v55 == &v55[(unsigned int)v56] )
            {
LABEL_108:
              sub_1DD6D30((__int64)v48, (_QWORD *)v58[v37]);
              v6 = v44;
            }
            else
            {
              while ( !sub_20D6290(*v36, v58[v37]) )
              {
                v36 = (__int64 *)(v38 + 8);
                if ( (__int64 *)v5 == v36 )
                  goto LABEL_108;
              }
            }
            ++v35;
          }
          while ( (unsigned int)v59 > v35 );
          v45 = v6;
        }
        if ( v58 != &v60 )
          _libc_free((unsigned __int64)v58);
LABEL_41:
        if ( v55 != (__int64 *)v57 )
          _libc_free((unsigned __int64)v55);
      }
LABEL_43:
      if ( v52 != (__int64 *)v54 )
        _libc_free((unsigned __int64)v52);
      if ( v49 != (__int64 *)v51 )
        _libc_free((unsigned __int64)v49);
LABEL_12:
      if ( (*(_BYTE *)v8 & 4) == 0 )
        break;
      v8 = *(_QWORD *)(v8 + 8);
      if ( v42 == (_QWORD *)v8 )
        goto LABEL_14;
    }
    while ( (*(_BYTE *)(v8 + 46) & 8) != 0 )
      v8 = *(_QWORD *)(v8 + 8);
    v8 = *(_QWORD *)(v8 + 8);
  }
  while ( v42 != (_QWORD *)v8 );
LABEL_14:
  if ( v45 )
  {
    v9 = *(_BYTE *)(a1 + 139);
    if ( v9 )
    {
      v10 = v48;
      v60 = 0x800000000LL;
      v58 = 0;
      v59 = (unsigned __int64)v61;
      v62 = 0;
      v63 = 0;
      sub_1DD77B0((__int64)v48);
      sub_1DC3250((__int64)&v58, v10);
      _libc_free(v62);
      if ( (_BYTE *)v59 != v61 )
        _libc_free(v59);
      v45 = v9;
    }
    goto LABEL_25;
  }
LABEL_24:
  v45 = 0;
LABEL_25:
  if ( v64 != v66 )
    _libc_free((unsigned __int64)v64);
  return v45;
}
