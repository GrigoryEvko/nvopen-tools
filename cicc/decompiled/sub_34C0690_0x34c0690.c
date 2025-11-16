// Function: sub_34C0690
// Address: 0x34c0690
//
__int64 __fastcall sub_34C0690(__int64 a1, __int64 a2)
{
  __int64 v2; // rdi
  __int64 (*v3)(); // rax
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD *v8; // rdx
  __int64 v9; // r14
  unsigned __int8 v10; // bl
  _QWORD *v11; // r13
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  _QWORD *v16; // rax
  _QWORD *v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r13
  __int64 v21; // rbx
  __int64 v22; // rax
  _QWORD *v23; // rax
  _QWORD *v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  __int64 *v29; // r12
  __int64 *v30; // rbx
  __int64 *v31; // r14
  __int64 v32; // r13
  __int64 v33; // r15
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // rdi
  _QWORD *v38; // rax
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  bool v41; // al
  unsigned __int64 v42; // r12
  __int64 *v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // rdx
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  __int64 v48; // [rsp+0h] [rbp-250h]
  __int64 *v49; // [rsp+18h] [rbp-238h]
  __int64 v50; // [rsp+38h] [rbp-218h]
  bool v52; // [rsp+56h] [rbp-1FAh]
  unsigned __int8 v53; // [rsp+57h] [rbp-1F9h]
  _QWORD *v54; // [rsp+58h] [rbp-1F8h]
  __int64 v55; // [rsp+60h] [rbp-1F0h] BYREF
  _QWORD *v56; // [rsp+68h] [rbp-1E8h] BYREF
  __int64 *v57; // [rsp+70h] [rbp-1E0h] BYREF
  __int64 v58; // [rsp+78h] [rbp-1D8h]
  _BYTE v59[48]; // [rsp+80h] [rbp-1D0h] BYREF
  __int64 *v60; // [rsp+B0h] [rbp-1A0h] BYREF
  __int64 v61; // [rsp+B8h] [rbp-198h]
  _BYTE v62[48]; // [rsp+C0h] [rbp-190h] BYREF
  __int64 *v63; // [rsp+F0h] [rbp-160h] BYREF
  __int64 v64; // [rsp+F8h] [rbp-158h]
  _QWORD v65[6]; // [rsp+100h] [rbp-150h] BYREF
  _QWORD *v66; // [rsp+130h] [rbp-120h] BYREF
  unsigned __int64 v67; // [rsp+138h] [rbp-118h]
  _QWORD v68[2]; // [rsp+140h] [rbp-110h] BYREF
  _BYTE v69[16]; // [rsp+150h] [rbp-100h] BYREF
  unsigned __int64 v70; // [rsp+160h] [rbp-F0h]
  int v71; // [rsp+168h] [rbp-E8h]
  _BYTE *v72; // [rsp+170h] [rbp-E0h] BYREF
  __int64 v73; // [rsp+178h] [rbp-D8h]
  _BYTE v74[208]; // [rsp+180h] [rbp-D0h] BYREF

  v2 = *(_QWORD *)(a1 + 136);
  v72 = v74;
  v55 = 0;
  v56 = 0;
  v73 = 0x400000000LL;
  v53 = 0;
  v3 = *(__int64 (**)())(*(_QWORD *)v2 + 344LL);
  if ( v3 == sub_2DB1AE0 )
    return v53;
  v53 = ((__int64 (__fastcall *)(__int64, __int64, __int64 *, _QWORD **, _BYTE **, __int64))v3)(
          v2,
          a2,
          &v55,
          &v56,
          &v72,
          1);
  if ( v53 || !v55 || !(_DWORD)v73 )
    goto LABEL_28;
  v8 = v56;
  if ( !v56 )
  {
    v16 = *(_QWORD **)(a2 + 112);
    v17 = &v16[*(unsigned int *)(a2 + 120)];
    if ( v17 == v16 )
      goto LABEL_28;
    while ( 1 )
    {
      v8 = (_QWORD *)*v16;
      if ( v55 != *v16 )
        break;
      if ( v17 == ++v16 )
        goto LABEL_28;
    }
    v56 = (_QWORD *)*v16;
    if ( !v8 )
      goto LABEL_28;
  }
  if ( *((_DWORD *)v8 + 18) > 1u )
    goto LABEL_28;
  v9 = *(_QWORD *)(a2 + 56);
  v50 = a2 + 48;
  if ( v9 == a2 + 48 )
    goto LABEL_28;
  v54 = (_QWORD *)a2;
  do
  {
    while ( 1 )
    {
      v52 = (*(_QWORD *)(*(_QWORD *)(v9 + 16) + 24LL) & 0x2000LL) != 0;
      if ( (*(_QWORD *)(*(_QWORD *)(v9 + 16) + 24LL) & 0x2000LL) == 0 )
        goto LABEL_12;
      v57 = (__int64 *)v59;
      v60 = (__int64 *)v62;
      v63 = v65;
      v58 = 0x600000000LL;
      v61 = 0x600000000LL;
      v64 = 0x600000000LL;
      v18 = *(unsigned int *)(*(_QWORD *)(v9 + 32) + 8LL);
      v19 = *(_QWORD *)(a1 + 144);
      if ( (int)v18 < 0 )
        v20 = *(_QWORD *)(*(_QWORD *)(v19 + 56) + 16 * (v18 & 0x7FFFFFFF) + 8);
      else
        v20 = *(_QWORD *)(*(_QWORD *)(v19 + 304) + 8 * v18);
      if ( v20 )
      {
        while ( (*(_BYTE *)(v20 + 4) & 8) != 0 )
        {
          v20 = *(_QWORD *)(v20 + 32);
          if ( !v20 )
            goto LABEL_47;
        }
LABEL_38:
        v21 = *(_QWORD *)(v20 + 16);
        if ( (*(_BYTE *)(v20 + 3) & 0x10) != 0 )
        {
          if ( !sub_2E88AF0(*(_QWORD *)(v20 + 16), v9, 1u) )
          {
            v23 = *(_QWORD **)(v21 + 24);
            if ( v54 == v23 )
            {
              if ( sub_34BE750(v9, v21) )
                goto LABEL_45;
              v23 = *(_QWORD **)(v21 + 24);
            }
            if ( v56 == v23 )
            {
              v27 = (unsigned int)v64;
              v28 = (unsigned int)v64 + 1LL;
              if ( v28 > HIDWORD(v64) )
              {
                sub_C8D5F0((__int64)&v63, v65, v28, 8u, v6, v7);
                v27 = (unsigned int)v64;
              }
              v63[v27] = v21;
              LODWORD(v64) = v64 + 1;
            }
          }
          if ( *(_WORD *)(v21 + 68) != 20 || (*(_BYTE *)(v20 + 3) & 0x10) != 0 )
            goto LABEL_42;
        }
        else if ( *(_WORD *)(v21 + 68) != 20 )
        {
          goto LABEL_42;
        }
        v24 = *(_QWORD **)(v21 + 24);
        if ( v54 != v24 )
          goto LABEL_60;
        if ( !sub_34BE750(v9, v21) )
        {
          v24 = *(_QWORD **)(v21 + 24);
LABEL_60:
          if ( v56 == v24 )
          {
            v25 = (unsigned int)v61;
            v26 = (unsigned int)v61 + 1LL;
            if ( v26 > HIDWORD(v61) )
            {
              sub_C8D5F0((__int64)&v60, v62, v26, 8u, v6, v7);
              v25 = (unsigned int)v61;
            }
            v60[v25] = v21;
            LODWORD(v61) = v61 + 1;
          }
          goto LABEL_42;
        }
        v46 = (unsigned int)v58;
        v47 = (unsigned int)v58 + 1LL;
        if ( v47 > HIDWORD(v58) )
        {
          sub_C8D5F0((__int64)&v57, v59, v47, 8u, v6, v7);
          v46 = (unsigned int)v58;
        }
        v57[v46] = v21;
        LODWORD(v58) = v58 + 1;
LABEL_42:
        while ( 1 )
        {
          v20 = *(_QWORD *)(v20 + 32);
          if ( !v20 )
            break;
          if ( (*(_BYTE *)(v20 + 4) & 8) == 0 )
            goto LABEL_38;
        }
        v22 = (unsigned int)v58;
        if ( !(_DWORD)v58 || !(_DWORD)v61 )
          goto LABEL_45;
        v48 = v9;
        v66 = v68;
        v29 = v60;
        v67 = 0x600000000LL;
        v49 = &v60[(unsigned int)v61];
        while ( 1 )
        {
          v6 = (__int64)v57;
          v30 = &v57[v22];
          v31 = v57;
          v32 = *v29;
          while ( v30 != v31 )
          {
            v33 = *v31;
            if ( sub_2E88AF0(*v31, v32, 1u) )
            {
              v34 = *(unsigned int *)(*(_QWORD *)(v33 + 32) + 8LL);
              v35 = *(_QWORD *)(a1 + 144);
              if ( (int)v34 < 0 )
                v36 = *(_QWORD *)(*(_QWORD *)(v35 + 56) + 16 * (v34 & 0x7FFFFFFF) + 8);
              else
                v36 = *(_QWORD *)(*(_QWORD *)(v35 + 304) + 8 * v34);
              if ( v36 )
              {
                if ( (*(_BYTE *)(v36 + 4) & 8) != 0 )
                {
                  while ( 1 )
                  {
                    v36 = *(_QWORD *)(v36 + 32);
                    if ( !v36 )
                      break;
                    if ( (*(_BYTE *)(v36 + 4) & 8) == 0 )
                      goto LABEL_82;
                  }
                }
                else
                {
LABEL_82:
                  if ( (*(_BYTE *)(v36 + 3) & 0x10) != 0 )
                  {
                    v37 = *(_QWORD *)(v36 + 16);
                    if ( v33 != v37 && v32 != v37 )
                    {
                      v38 = *(_QWORD **)(v37 + 24);
                      if ( v54 == v38 )
                      {
                        v41 = sub_34BE750(v33, *(_QWORD *)(v36 + 16));
                      }
                      else
                      {
                        if ( v56 != v38 )
                          goto LABEL_89;
                        v41 = sub_34BE750(v37, v32);
                      }
                      if ( v41 )
                        goto LABEL_75;
                    }
                  }
LABEL_89:
                  while ( 1 )
                  {
                    v36 = *(_QWORD *)(v36 + 32);
                    if ( !v36 )
                      break;
                    if ( (*(_BYTE *)(v36 + 4) & 8) == 0 )
                      goto LABEL_82;
                  }
                }
              }
              v39 = (unsigned int)v67;
              v40 = (unsigned int)v67 + 1LL;
              if ( v40 > HIDWORD(v67) )
              {
                sub_C8D5F0((__int64)&v66, v68, v40, 8u, v6, v7);
                v39 = (unsigned int)v67;
              }
              v66[v39] = v32;
              LODWORD(v67) = v67 + 1;
            }
LABEL_75:
            ++v31;
          }
          if ( v49 == ++v29 )
            break;
          v22 = (unsigned int)v58;
        }
        v9 = v48;
        if ( (_DWORD)v67 )
        {
          v42 = 0;
          do
          {
            v43 = v63;
            v44 = v42;
            if ( &v63[(unsigned int)v64] == v63 )
            {
LABEL_110:
              sub_2E325D0((__int64)v56, (_QWORD *)v66[v44]);
              v7 = v52;
            }
            else
            {
              while ( !sub_34BE750(*v43, v66[v44]) )
              {
                v43 = (__int64 *)(v45 + 8);
                if ( (__int64 *)v6 == v43 )
                  goto LABEL_110;
              }
            }
            ++v42;
          }
          while ( v42 < (unsigned int)v67 );
          v53 = v7;
        }
        if ( v66 != v68 )
          _libc_free((unsigned __int64)v66);
LABEL_45:
        if ( v63 != v65 )
          _libc_free((unsigned __int64)v63);
      }
LABEL_47:
      if ( v60 != (__int64 *)v62 )
        _libc_free((unsigned __int64)v60);
      if ( v57 != (__int64 *)v59 )
        _libc_free((unsigned __int64)v57);
LABEL_12:
      if ( (*(_BYTE *)v9 & 4) == 0 )
        break;
      v9 = *(_QWORD *)(v9 + 8);
      if ( v50 == v9 )
        goto LABEL_14;
    }
    while ( (*(_BYTE *)(v9 + 44) & 8) != 0 )
      v9 = *(_QWORD *)(v9 + 8);
    v9 = *(_QWORD *)(v9 + 8);
  }
  while ( v50 != v9 );
LABEL_14:
  if ( v53 )
  {
    v10 = *(_BYTE *)(a1 + 131);
    if ( v10 )
    {
      v11 = v56;
      v66 = 0;
      v68[0] = 0;
      v67 = (unsigned __int64)v69;
      v68[1] = 8;
      v70 = 0;
      v71 = 0;
      v63 = 0;
      v64 = 0;
      v65[0] = 0;
      sub_2E330F0(v56, &v63);
      sub_3509790(&v66, v11);
      sub_2E31EE0((__int64)v11, (__int64)v11, v12, v13, v14, v15);
      if ( v63 )
        j_j___libc_free_0((unsigned __int64)v63);
      if ( v70 )
        _libc_free(v70);
      if ( (_BYTE *)v67 != v69 )
        _libc_free(v67);
      v53 = v10;
    }
    goto LABEL_29;
  }
LABEL_28:
  v53 = 0;
LABEL_29:
  if ( v72 != v74 )
    _libc_free((unsigned __int64)v72);
  return v53;
}
