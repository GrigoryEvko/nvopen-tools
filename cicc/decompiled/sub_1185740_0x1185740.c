// Function: sub_1185740
// Address: 0x1185740
//
unsigned __int8 *__fastcall sub_1185740(__int64 a1, unsigned int **a2)
{
  __int64 v4; // rbx
  bool v5; // zf
  __int64 v6; // rdi
  unsigned __int8 v7; // al
  __int64 v8; // rdi
  __int64 v9; // r14
  __int64 v10; // rdi
  bool v11; // bl
  _BYTE *v12; // rdi
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int8 *v16; // r12
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // rcx
  _BYTE *v21; // rdi
  char v22; // al
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 *v26; // rdx
  unsigned __int8 **v27; // rdx
  unsigned __int8 *v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // r12
  __int64 *v33; // rdx
  __int64 v34; // rdi
  __int64 v35; // rcx
  __int64 v36; // rsi
  __int64 v37; // rbx
  __int64 v38; // r13
  unsigned __int8 *v39; // r14
  unsigned __int8 *v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rdi
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // rdx
  __int64 v51; // rbx
  __int64 v52; // rdx
  __int64 v53; // rsi
  __int64 *v54; // rax
  __int64 v55; // rcx
  __int64 *v56; // rax
  __int64 v57; // rcx
  _BYTE *v58; // rdi
  _BYTE *v59; // rdi
  __int64 v60; // rbx
  __int64 v61; // [rsp+0h] [rbp-C0h]
  __int64 v62; // [rsp+0h] [rbp-C0h]
  unsigned __int8 *v63; // [rsp+8h] [rbp-B8h]
  __int64 v64; // [rsp+8h] [rbp-B8h]
  __int64 v65; // [rsp+8h] [rbp-B8h]
  __int64 v66; // [rsp+10h] [rbp-B0h] BYREF
  unsigned __int8 *v67; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v68; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v69; // [rsp+28h] [rbp-98h]
  __int64 v70; // [rsp+30h] [rbp-90h]
  unsigned __int8 *v71; // [rsp+40h] [rbp-80h] BYREF
  __int64 v72; // [rsp+48h] [rbp-78h]
  __int64 v73; // [rsp+50h] [rbp-70h]
  _QWORD *v74; // [rsp+60h] [rbp-60h] BYREF
  __int64 *v75; // [rsp+68h] [rbp-58h]
  unsigned __int8 **v76; // [rsp+70h] [rbp-50h]
  __int64 *v77; // [rsp+78h] [rbp-48h]
  __int16 v78; // [rsp+80h] [rbp-40h]

  v4 = 0;
  v5 = *(_BYTE *)a1 == 86;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  if ( v5 )
  {
    if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    {
      v26 = *(__int64 **)(a1 - 8);
      v4 = *v26;
      if ( !*v26 )
        goto LABEL_2;
    }
    else
    {
      v26 = (__int64 *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
      v4 = *v26;
      if ( !*v26 )
        goto LABEL_2;
    }
    v68 = v4;
    if ( v26[4] )
    {
      v69 = v26[4];
      if ( v26[8] )
        v70 = v26[8];
    }
  }
LABEL_2:
  v74 = 0;
  v75 = &v68;
  if ( *(_BYTE *)v4 == 59 )
  {
    v22 = sub_995B10(&v74, *(_QWORD *)(v4 - 64));
    v23 = *(_QWORD *)(v4 - 32);
    if ( v22 && v23 )
    {
      *v75 = v23;
    }
    else
    {
      if ( !(unsigned __int8)sub_995B10(&v74, v23) )
        goto LABEL_45;
      v24 = *(_QWORD *)(v4 - 64);
      if ( !v24 )
        goto LABEL_45;
      *v75 = v24;
    }
    v25 = v70;
    v70 = v69;
    v69 = v25;
LABEL_45:
    v4 = v68;
  }
  if ( *(_BYTE *)v4 <= 0x1Cu )
    return 0;
  v6 = *(_QWORD *)(v4 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 <= 1 )
    v6 = **(_QWORD **)(v6 + 16);
  v5 = !sub_BCAC40(v6, 1);
  v7 = *(_BYTE *)v4;
  if ( v5 )
    goto LABEL_26;
  if ( v7 == 57 )
    goto LABEL_12;
  if ( v7 != 86 )
    goto LABEL_26;
  v8 = *(_QWORD *)(v4 + 8);
  if ( *(_QWORD *)(*(_QWORD *)(v4 - 96) + 8LL) == v8 && **(_BYTE **)(v4 - 32) <= 0x15u )
  {
    if ( sub_AC30F0(*(_QWORD *)(v4 - 32)) )
      goto LABEL_12;
    v7 = *(_BYTE *)v4;
LABEL_26:
    if ( v7 <= 0x1Cu )
      return 0;
    v8 = *(_QWORD *)(v4 + 8);
  }
  if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
    v8 = **(_QWORD **)(v8 + 16);
  if ( !sub_BCAC40(v8, 1) )
    return 0;
  if ( *(_BYTE *)v4 != 58 )
  {
    if ( *(_BYTE *)v4 != 86 )
      return 0;
    v20 = *(_QWORD *)(v4 + 8);
    if ( *(_QWORD *)(*(_QWORD *)(v4 - 96) + 8LL) != v20 )
      return 0;
    v21 = *(_BYTE **)(v4 - 64);
    if ( *v21 > 0x15u || !sub_AD7A80(v21, 1, v18, v20, v19) )
      return 0;
  }
LABEL_12:
  v9 = v68;
  if ( *(_BYTE *)v68 > 0x1Cu )
  {
    v10 = *(_QWORD *)(v68 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 <= 1 )
      v10 = **(_QWORD **)(v10 + 16);
    v11 = sub_BCAC40(v10, 1);
    if ( v11 )
    {
      if ( *(_BYTE *)v9 == 57
        || *(_BYTE *)v9 == 86
        && *(_QWORD *)(*(_QWORD *)(v9 - 96) + 8LL) == *(_QWORD *)(v9 + 8)
        && (v12 = *(_BYTE **)(v9 - 32), *v12 <= 0x15u)
        && sub_AC30F0((__int64)v12) )
      {
        v13 = v70;
        v14 = *(_QWORD *)(*(_QWORD *)(a1 - 96) + 16LL);
        if ( !v14 )
          goto LABEL_22;
LABEL_38:
        if ( !*(_QWORD *)(v14 + 8) )
          goto LABEL_56;
        goto LABEL_22;
      }
    }
  }
  v13 = v69;
  v11 = 0;
  v14 = *(_QWORD *)(*(_QWORD *)(a1 - 96) + 16LL);
  if ( v14 )
    goto LABEL_38;
LABEL_22:
  v15 = *(_QWORD *)(v13 + 16);
  if ( !v15 || *(_QWORD *)(v15 + 8) )
    return 0;
LABEL_56:
  v71 = 0;
  v72 = 0;
  v73 = 0;
  if ( *(_BYTE *)v13 != 86 )
    return 0;
  v27 = (*(_BYTE *)(v13 + 7) & 0x40) != 0
      ? *(unsigned __int8 ***)(v13 - 8)
      : (unsigned __int8 **)(v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF));
  v28 = *v27;
  if ( !v28 )
    return 0;
  v71 = v28;
  v29 = (*(_BYTE *)(v13 + 7) & 0x40) != 0 ? *(_QWORD *)(v13 - 8) : v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF);
  if ( !*(_QWORD *)(v29 + 32) )
    return 0;
  v72 = *(_QWORD *)(v29 + 32);
  v30 = (*(_BYTE *)(v13 + 7) & 0x40) != 0 ? *(_QWORD *)(v13 - 8) : v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF);
  if ( !*(_QWORD *)(v30 + 64) )
    return 0;
  v73 = *(_QWORD *)(v30 + 64);
  v74 = 0;
  v75 = (__int64 *)&v71;
  if ( (unsigned __int8)sub_996420(&v74, 30, v28) )
  {
    v31 = v73;
    v73 = v72;
    v72 = v31;
  }
  v66 = 0;
  v32 = v68;
  v33 = (__int64 *)v71;
  if ( v11 )
  {
    if ( *(_BYTE *)v68 > 0x1Cu )
    {
      v34 = *(_QWORD *)(v68 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v34 + 8) - 17 <= 1 )
        v34 = **(_QWORD **)(v34 + 16);
      v63 = v71;
      if ( sub_BCAC40(v34, 1) )
      {
        if ( *(_BYTE *)v32 == 57 )
        {
          if ( (*(_BYTE *)(v32 + 7) & 0x40) != 0 )
            v56 = *(__int64 **)(v32 - 8);
          else
            v56 = (__int64 *)(v32 - 32LL * (*(_DWORD *)(v32 + 4) & 0x7FFFFFF));
          v36 = v56[4];
          v57 = *v56;
          if ( v36 && v63 == (unsigned __int8 *)v57 )
            goto LABEL_142;
          if ( v63 == (unsigned __int8 *)v36 && v57 )
          {
            v66 = *v56;
            v36 = v57;
LABEL_130:
            v35 = v72;
            v78 = 257;
            goto LABEL_79;
          }
        }
        else if ( *(_BYTE *)v32 == 86 && *(_QWORD *)(*(_QWORD *)(v32 - 96) + 8LL) == *(_QWORD *)(v32 + 8) )
        {
          v59 = *(_BYTE **)(v32 - 32);
          if ( *v59 <= 0x15u )
          {
            v62 = *(_QWORD *)(v32 - 96);
            v60 = *(_QWORD *)(v32 - 64);
            v36 = v62;
            if ( sub_AC30F0((__int64)v59) )
            {
              if ( v63 == (unsigned __int8 *)v62 )
              {
                if ( v60 )
                {
                  v66 = v60;
                  v36 = v60;
                  goto LABEL_130;
                }
              }
              else if ( v63 == (unsigned __int8 *)v60 )
              {
LABEL_142:
                v66 = v36;
                goto LABEL_130;
              }
            }
          }
        }
      }
      v33 = (__int64 *)v71;
    }
    v74 = 0;
    v76 = &v67;
    v75 = v33;
    v77 = &v66;
    if ( (unsigned __int8)sub_1185380((__int64)&v74, v32) )
    {
      v35 = v73;
      v78 = 257;
      v36 = v66;
      v73 = v72;
      v72 = v35;
      v71 = v67;
LABEL_79:
      v37 = sub_B36550(a2, v36, v69, v35, (__int64)&v74, 0);
      sub_BD6B90((unsigned __int8 *)v37, (unsigned __int8 *)v13);
      v38 = v73;
      v78 = 257;
      goto LABEL_80;
    }
    return 0;
  }
  if ( *(_BYTE *)v68 <= 0x1Cu )
    goto LABEL_110;
  v47 = *(_QWORD *)(v68 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v47 + 8) - 17 <= 1 )
    v47 = **(_QWORD **)(v47 + 16);
  v64 = (__int64)v71;
  if ( sub_BCAC40(v47, 1) )
  {
    v50 = v64;
    if ( *(_BYTE *)v32 == 58 )
    {
      if ( (*(_BYTE *)(v32 + 7) & 0x40) != 0 )
        v54 = *(__int64 **)(v32 - 8);
      else
        v54 = (__int64 *)(v32 - 32LL * (*(_DWORD *)(v32 + 4) & 0x7FFFFFF));
      v55 = *v54;
      v53 = v54[4];
      if ( v64 != *v54 || !v53 )
      {
        if ( v64 == v53 && v55 )
        {
          v66 = *v54;
          v53 = v55;
LABEL_122:
          v52 = v73;
          v78 = 257;
          goto LABEL_112;
        }
        goto LABEL_109;
      }
      goto LABEL_144;
    }
    if ( *(_BYTE *)v32 == 86 )
    {
      v51 = *(_QWORD *)(v32 - 96);
      if ( *(_QWORD *)(v51 + 8) == *(_QWORD *)(v32 + 8) )
      {
        v58 = *(_BYTE **)(v32 - 64);
        if ( *v58 <= 0x15u )
        {
          v61 = v64;
          v65 = *(_QWORD *)(v32 - 32);
          v53 = v65;
          if ( sub_AD7A80(v58, v65, v50, v48, v49) )
          {
            if ( v61 != v51 )
            {
              if ( v61 == v65 )
              {
                v66 = v51;
                v53 = v51;
                goto LABEL_122;
              }
              goto LABEL_109;
            }
            if ( v65 )
            {
LABEL_144:
              v66 = v53;
              goto LABEL_122;
            }
          }
        }
      }
    }
  }
LABEL_109:
  v33 = (__int64 *)v71;
LABEL_110:
  v74 = 0;
  v76 = &v67;
  v75 = v33;
  v77 = &v66;
  if ( !(unsigned __int8)sub_1185560((__int64)&v74, v32) )
    return 0;
  v52 = v72;
  v53 = v66;
  v72 = v73;
  v73 = v52;
  v71 = v67;
  v78 = 257;
LABEL_112:
  v38 = sub_B36550(a2, v53, v52, v70, (__int64)&v74, 0);
  sub_BD6B90((unsigned __int8 *)v38, (unsigned __int8 *)v13);
  v37 = v72;
  v78 = 257;
LABEL_80:
  v39 = v71;
  v40 = (unsigned __int8 *)sub_BD2C40(72, 3u);
  v16 = v40;
  if ( v40 )
  {
    sub_B44260((__int64)v40, *(_QWORD *)(v37 + 8), 57, 3u, 0, 0);
    if ( *((_QWORD *)v16 - 12) )
    {
      v41 = *((_QWORD *)v16 - 11);
      **((_QWORD **)v16 - 10) = v41;
      if ( v41 )
        *(_QWORD *)(v41 + 16) = *((_QWORD *)v16 - 10);
    }
    *((_QWORD *)v16 - 12) = v39;
    if ( v39 )
    {
      v42 = *((_QWORD *)v39 + 2);
      *((_QWORD *)v16 - 11) = v42;
      if ( v42 )
        *(_QWORD *)(v42 + 16) = v16 - 88;
      *((_QWORD *)v16 - 10) = v39 + 16;
      *((_QWORD *)v39 + 2) = v16 - 96;
    }
    if ( *((_QWORD *)v16 - 8) )
    {
      v43 = *((_QWORD *)v16 - 7);
      **((_QWORD **)v16 - 6) = v43;
      if ( v43 )
        *(_QWORD *)(v43 + 16) = *((_QWORD *)v16 - 6);
    }
    *((_QWORD *)v16 - 8) = v37;
    v44 = *(_QWORD *)(v37 + 16);
    *((_QWORD *)v16 - 7) = v44;
    if ( v44 )
      *(_QWORD *)(v44 + 16) = v16 - 56;
    *((_QWORD *)v16 - 6) = v37 + 16;
    *(_QWORD *)(v37 + 16) = v16 - 64;
    if ( *((_QWORD *)v16 - 4) )
    {
      v45 = *((_QWORD *)v16 - 3);
      **((_QWORD **)v16 - 2) = v45;
      if ( v45 )
        *(_QWORD *)(v45 + 16) = *((_QWORD *)v16 - 2);
    }
    *((_QWORD *)v16 - 4) = v38;
    if ( v38 )
    {
      v46 = *(_QWORD *)(v38 + 16);
      *((_QWORD *)v16 - 3) = v46;
      if ( v46 )
        *(_QWORD *)(v46 + 16) = v16 - 24;
      *((_QWORD *)v16 - 2) = v38 + 16;
      *(_QWORD *)(v38 + 16) = v16 - 32;
    }
    sub_BD6B50(v16, (const char **)&v74);
  }
  return v16;
}
