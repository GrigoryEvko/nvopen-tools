// Function: sub_2B65070
// Address: 0x2b65070
//
__int64 __fastcall sub_2B65070(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 *a4,
        __int64 a5,
        int a6,
        void *src,
        __int64 a8)
{
  __int64 v11; // rdi
  __int64 result; // rax
  unsigned __int8 *v13; // r11
  __int64 v14; // r10
  unsigned __int8 v15; // al
  unsigned __int8 v16; // r12
  __int64 v17; // r9
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // r9
  unsigned int v23; // edx
  __int64 v24; // rax
  unsigned __int64 *v25; // r10
  __int64 v26; // rax
  unsigned __int64 *v27; // rsi
  unsigned __int64 *v28; // rax
  unsigned __int8 *v29; // r12
  __int64 v30; // r13
  unsigned __int64 v31; // rcx
  unsigned __int64 v32; // rcx
  unsigned __int64 v33; // rcx
  signed __int64 v34; // rcx
  char **v35; // rax
  char *v36; // r8
  char *v37; // rcx
  __int64 v38; // rdi
  __int64 *v39; // rax
  __int64 v40; // r15
  __int64 v41; // r14
  char v42; // al
  _QWORD *v43; // rax
  _QWORD *v44; // rcx
  int v45; // ecx
  __int64 *v46; // rdi
  int v47; // eax
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 v52; // rdi
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // rdi
  char *v58; // rsi
  __int64 *v59; // rdi
  char v60; // bl
  __int64 v61; // rsi
  unsigned __int64 *v62; // rdi
  __int64 v63; // [rsp+8h] [rbp-138h]
  int v64; // [rsp+8h] [rbp-138h]
  unsigned __int8 *v65; // [rsp+10h] [rbp-130h]
  __int64 v66; // [rsp+10h] [rbp-130h]
  char *v67; // [rsp+10h] [rbp-130h]
  unsigned __int8 *v68; // [rsp+10h] [rbp-130h]
  __int64 v69; // [rsp+10h] [rbp-130h]
  __int64 v70; // [rsp+18h] [rbp-128h]
  __int64 v71; // [rsp+18h] [rbp-128h]
  unsigned int v72; // [rsp+18h] [rbp-128h]
  unsigned __int8 *v73; // [rsp+18h] [rbp-128h]
  char *v74; // [rsp+18h] [rbp-128h]
  unsigned int v75; // [rsp+18h] [rbp-128h]
  unsigned __int8 *v76; // [rsp+18h] [rbp-128h]
  __int64 v77; // [rsp+20h] [rbp-120h] BYREF
  unsigned __int8 *v78; // [rsp+28h] [rbp-118h] BYREF
  _QWORD v79[2]; // [rsp+30h] [rbp-110h] BYREF
  __int64 *v80[4]; // [rsp+40h] [rbp-100h] BYREF
  __int64 v81; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v82; // [rsp+68h] [rbp-D8h]
  _QWORD v83[8]; // [rsp+70h] [rbp-D0h] BYREF
  unsigned __int64 *v84; // [rsp+B0h] [rbp-90h] BYREF
  unsigned __int64 v85; // [rsp+B8h] [rbp-88h]
  __int64 v86; // [rsp+C0h] [rbp-80h] BYREF
  int v87; // [rsp+C8h] [rbp-78h]
  char v88; // [rsp+CCh] [rbp-74h]
  char v89; // [rsp+D0h] [rbp-70h] BYREF

  v11 = *(_QWORD *)(a2 + 8);
  v78 = (unsigned __int8 *)a2;
  v77 = a3;
  if ( !sub_2B08630(v11) || !sub_2B08630(*(_QWORD *)(v77 + 8)) )
    return 0;
  v13 = v78;
  v14 = v77;
  if ( v78 == (unsigned __int8 *)v77 )
  {
    if ( *v78 == 61 )
    {
      v46 = *(__int64 **)(*(_QWORD *)(a1 + 24) + 3296LL);
      v47 = *(_DWORD *)(a1 + 32);
      BYTE4(v80[0]) = 0;
      LODWORD(v80[0]) = v47;
      if ( (unsigned __int8)sub_DFA4E0(v46, *((_QWORD *)v78 + 1), (__int64)v80[0]) )
      {
        if ( (unsigned int)sub_BD3960((__int64)v78) == *(_DWORD *)(a1 + 32) )
          return 3;
        if ( !(unsigned __int8)sub_BD3660((__int64)v78, 64) && !(unsigned __int8)sub_BD3660(v77, 64) )
        {
          v52 = *((_QWORD *)v78 + 2);
          v82 = a5;
          v83[0] = a1;
          v81 = (__int64)a4;
          if ( sub_2B149F0(v52, 0, v48, v49, v50, v51, (__int64)a4, a5, a1) )
          {
            v57 = *(_QWORD *)(v77 + 16);
            v85 = a5;
            v86 = a1;
            v84 = a4;
            if ( sub_2B149F0(v57, 0, v53, v54, v55, v56, (__int64)a4, a5, a1) )
              return 3;
          }
        }
      }
    }
    return 1;
  }
  v80[0] = (__int64 *)a1;
  v80[1] = (__int64 *)&v78;
  v80[2] = &v77;
  v15 = *v78;
  v16 = *(_BYTE *)v77;
  if ( *v78 == 61 )
  {
    if ( v16 != 61 )
      goto LABEL_8;
    v71 = v77;
    if ( *((_QWORD *)v78 + 5) != *(_QWORD *)(v77 + 40) )
      return sub_2B2A1B0(v80);
    v65 = v78;
    if ( sub_B46500(v78) || (v65[2] & 1) != 0 || sub_B46500((unsigned __int8 *)v71) || (*(_BYTE *)(v71 + 2) & 1) != 0 )
      return sub_2B2A1B0(v80);
    v84 = (unsigned __int64 *)sub_D35010(
                                *((_QWORD *)v65 + 1),
                                *((_QWORD *)v65 - 4),
                                *(_QWORD *)(v71 + 8),
                                *(_QWORD *)(v71 - 32),
                                *(_QWORD *)(a1 + 8),
                                *(_QWORD *)(a1 + 16),
                                1,
                                1);
    if ( BYTE4(v84) == 1 && (_DWORD)v84 )
    {
      if ( (int)abs32((int)v84) <= *(_DWORD *)(a1 + 32) / 2 )
        return (unsigned int)((int)v84 > 0) + 3;
    }
    else
    {
      v29 = sub_98ACB0(*((unsigned __int8 **)v65 - 4), 6u);
      if ( v29 != sub_98ACB0(*(unsigned __int8 **)(v71 - 32), 6u) )
        return sub_2B2A1B0(v80);
      v30 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 3296LL);
      sub_2B08680(*((_QWORD *)v65 + 1), *(_DWORD *)(a1 + 32));
      if ( !(unsigned __int8)sub_DFA510(v30) )
        return sub_2B2A1B0(v80);
    }
    return 1;
  }
  if ( v15 <= 0x15u )
  {
    if ( v16 <= 0x15u )
      return 2;
    goto LABEL_30;
  }
  if ( v15 != 90 )
  {
LABEL_30:
    if ( v15 <= 0x1Cu )
    {
LABEL_26:
      if ( (unsigned int)v16 - 12 > 1 )
        return sub_2B2A1B0(v80);
      return 1;
    }
LABEL_8:
    if ( v16 <= 0x1Cu )
      goto LABEL_25;
    if ( *(_QWORD *)(v14 + 40) != *((_QWORD *)v13 + 5) )
      return sub_2B2A1B0(v80);
    v17 = 8 * a8;
    v84 = (unsigned __int64 *)&v86;
    v85 = 0x400000000LL;
    v18 = (8 * a8) >> 3;
    if ( (unsigned __int64)(8 * a8) > 0x20 )
    {
      v63 = v14;
      v68 = v13;
      sub_C8D5F0((__int64)&v84, &v86, (8 * a8) >> 3, 8u, v18, v17);
      v18 = (8 * a8) >> 3;
      v13 = v68;
      v14 = v63;
      v17 = 8 * a8;
      v59 = (__int64 *)&v84[(unsigned int)v85];
    }
    else
    {
      if ( !v17 )
      {
LABEL_12:
        v70 = v14;
        LODWORD(v85) = v17 + v18;
        sub_94F890((__int64)&v84, (__int64)v13);
        sub_94F890((__int64)&v84, v70);
        v19 = sub_2B5F980((__int64 *)v84, (unsigned int)v85, *(__int64 **)a1);
        v21 = v19;
        v22 = v20;
        if ( !v19 || !v20 || (v23 = *(_DWORD *)(v19 + 4) & 0x7FFFFFF, v23 > 2) && !a8 && v19 != v22 )
        {
LABEL_22:
          if ( v84 != (unsigned __int64 *)&v86 )
            _libc_free((unsigned __int64)v84);
          v16 = *(_BYTE *)v77;
LABEL_25:
          if ( v16 != 13 )
            goto LABEL_26;
          return 2;
        }
        v24 = (unsigned int)v85;
        v25 = &v84[v24];
        v26 = (v24 * 8) >> 5;
        if ( v26 )
        {
          v27 = &v84[4 * v26];
          v28 = v84;
          while ( *(_BYTE *)*v28 == 13 || v23 == (*(_DWORD *)(*v28 + 4) & 0x7FFFFFF) )
          {
            v31 = v28[1];
            if ( *(_BYTE *)v31 != 13 && v23 != (*(_DWORD *)(v31 + 4) & 0x7FFFFFF) )
            {
              ++v28;
              break;
            }
            v32 = v28[2];
            if ( *(_BYTE *)v32 != 13 && v23 != (*(_DWORD *)(v32 + 4) & 0x7FFFFFF) )
            {
              v28 += 2;
              break;
            }
            v33 = v28[3];
            if ( *(_BYTE *)v33 != 13 && v23 != (*(_DWORD *)(v33 + 4) & 0x7FFFFFF) )
            {
              v28 += 3;
              break;
            }
            v28 += 4;
            if ( v27 == v28 )
              goto LABEL_55;
          }
LABEL_21:
          if ( v25 != v28 )
            goto LABEL_22;
          goto LABEL_58;
        }
        v28 = v84;
LABEL_55:
        v34 = (char *)v25 - (char *)v28;
        if ( (char *)v25 - (char *)v28 != 16 )
        {
          if ( v34 != 24 )
          {
            if ( v34 != 8 )
              goto LABEL_58;
            goto LABEL_110;
          }
          if ( *(_BYTE *)*v28 != 13 && v23 != (*(_DWORD *)(*v28 + 4) & 0x7FFFFFF) )
            goto LABEL_21;
          ++v28;
        }
        if ( *(_BYTE *)*v28 != 13 && v23 != (*(_DWORD *)(*v28 + 4) & 0x7FFFFFF) )
          goto LABEL_21;
        ++v28;
LABEL_110:
        if ( *(_BYTE *)*v28 != 13 && v23 != (*(_DWORD *)(*v28 + 4) & 0x7FFFFFF) )
          goto LABEL_21;
LABEL_58:
        result = (unsigned int)(v21 == v22) + 1;
        if ( v84 != (unsigned __int64 *)&v86 )
        {
          v72 = (v21 == v22) + 1;
          _libc_free((unsigned __int64)v84);
          return v72;
        }
        return result;
      }
      v59 = &v86;
    }
    v64 = v18;
    v69 = v14;
    v76 = v13;
    memcpy(v59, src, v17);
    LODWORD(v17) = v85;
    LODWORD(v18) = v64;
    v14 = v69;
    v13 = v76;
    goto LABEL_12;
  }
  v66 = v77;
  v73 = v78;
  v35 = (char **)sub_986520((__int64)v78);
  v13 = v73;
  v14 = v66;
  v36 = *v35;
  if ( !*v35 )
    goto LABEL_8;
  v37 = v35[4];
  if ( *v37 != 17 )
    goto LABEL_8;
  if ( v16 == 12 )
  {
    v58 = *v35;
    v81 = 1;
    sub_2B25A00(&v84, v58, (unsigned __int64 *)&v81);
    v75 = (unsigned __int8)sub_2B0D9E0((unsigned __int64)v84) == 0 ? 2 : 4;
    sub_228BF40(&v84);
    sub_228BF40((unsigned __int64 **)&v81);
    return v75;
  }
  result = 4;
  if ( v16 != 13 )
  {
    if ( v16 != 90 )
      return sub_2B2A1B0(v80);
    v38 = v66;
    v67 = v37;
    v74 = v36;
    v39 = (__int64 *)sub_986520(v38);
    v40 = *v39;
    if ( !*v39 )
      return sub_2B2A1B0(v80);
    v41 = v39[4];
    v42 = *(_BYTE *)v41;
    if ( *(_BYTE *)v41 == 17 )
    {
      v81 = 1;
      sub_2B25A00(&v84, (char *)v40, (unsigned __int64 *)&v81);
      if ( !(unsigned __int8)sub_2B0D9E0((unsigned __int64)v84) || *((_QWORD *)v74 + 1) != *(_QWORD *)(v40 + 8) )
      {
        sub_228BF40(&v84);
        sub_228BF40((unsigned __int64 **)&v81);
        if ( v74 != (char *)v40 )
          return 1;
        v43 = (_QWORD *)*((_QWORD *)v67 + 3);
        if ( *((_DWORD *)v67 + 8) > 0x40u )
          v43 = (_QWORD *)*v43;
        v44 = *(_QWORD **)(v41 + 24);
        if ( *(_DWORD *)(v41 + 32) > 0x40u )
          v44 = (_QWORD *)*v44;
        v45 = (_DWORD)v44 - (_DWORD)v43;
        if ( !v45 )
          return 1;
        result = 2;
        if ( (int)abs32(v45) <= *(_DWORD *)(a1 + 32) / 2 )
          return (unsigned int)(v45 > 0) + 3;
        return result;
      }
      sub_228BF40(&v84);
      sub_228BF40((unsigned __int64 **)&v81);
    }
    else if ( (unsigned __int8)(v42 - 12) > 1u )
    {
      if ( (unsigned __int8)(v42 - 9) > 2u )
        return sub_2B2A1B0(v80);
      v88 = 1;
      v85 = (unsigned __int64)&v89;
      v82 = 0x800000000LL;
      v84 = 0;
      v86 = 8;
      v87 = 0;
      v81 = (__int64)v83;
      v79[0] = &v84;
      v79[1] = &v81;
      v60 = sub_AA8FD0(v79, v41);
      if ( v60 )
      {
        while ( 1 )
        {
          v62 = (unsigned __int64 *)v81;
          if ( !(_DWORD)v82 )
            break;
          v61 = *(_QWORD *)(v81 + 8LL * (unsigned int)v82 - 8);
          LODWORD(v82) = v82 - 1;
          if ( !(unsigned __int8)sub_AA8FD0(v79, v61) )
            goto LABEL_115;
        }
      }
      else
      {
LABEL_115:
        v62 = (unsigned __int64 *)v81;
        v60 = 0;
      }
      if ( v62 != v83 )
        _libc_free((unsigned __int64)v62);
      if ( !v88 )
        _libc_free(v85);
      if ( !v60 )
        return sub_2B2A1B0(v80);
    }
    return 4;
  }
  return result;
}
