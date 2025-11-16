// Function: sub_179DF30
// Address: 0x179df30
//
__int64 __fastcall sub_179DF30(
        __int64 a1,
        unsigned int a2,
        unsigned __int8 a3,
        __int64 *a4,
        __int64 a5,
        double a6,
        double a7,
        double a8)
{
  __int64 v12; // rbx
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned int v21; // r9d
  int v22; // ebx
  unsigned int v23; // eax
  __int64 **v24; // r15
  unsigned int v25; // eax
  __int64 v26; // rdx
  __int64 v27; // rcx
  _BYTE *v28; // rdi
  unsigned int v29; // r9d
  unsigned int v30; // r10d
  unsigned __int8 v31; // al
  _QWORD *v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rsi
  unsigned __int64 v36; // rdx
  __int64 v37; // rdx
  _QWORD *v38; // rax
  __int64 v39; // rax
  __int64 *v40; // rcx
  __int64 v41; // rsi
  unsigned __int64 v42; // rdx
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rdx
  _QWORD *v46; // rax
  __int64 v47; // rsi
  unsigned __int64 v48; // rcx
  __int64 v49; // rcx
  __int64 v50; // rcx
  _QWORD *v51; // rax
  __int64 v52; // rax
  __int64 v53; // rcx
  _QWORD *v54; // rax
  __int64 v55; // rsi
  unsigned __int64 v56; // rdx
  __int64 v57; // rdx
  __int64 v58; // rdx
  _QWORD *v59; // rax
  __int64 v60; // rax
  __int64 v61; // rdx
  _QWORD *v62; // rax
  __int64 v63; // rsi
  unsigned __int64 v64; // rcx
  __int64 v65; // rcx
  __int64 v66; // rcx
  _QWORD *v67; // rax
  __int64 v68; // rbx
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rax
  __int64 *v73; // rax
  __int64 v74; // rsi
  unsigned __int64 v75; // rcx
  __int64 v76; // rcx
  __int64 v77; // rax
  unsigned int v78; // eax
  unsigned int v79; // r14d
  unsigned __int64 v80; // rax
  __int64 v81; // rax
  unsigned __int8 *v82; // rax
  __int64 v83; // rax
  __int64 v84; // rsi
  unsigned __int64 v85; // rdx
  __int64 v86; // rdx
  unsigned int v87; // ebx
  __int64 v88; // [rsp+8h] [rbp-78h]
  _BYTE *v89; // [rsp+10h] [rbp-70h]
  __int64 v90; // [rsp+10h] [rbp-70h]
  char v91; // [rsp+10h] [rbp-70h]
  unsigned int v92; // [rsp+18h] [rbp-68h]
  unsigned int v94; // [rsp+1Ch] [rbp-64h]
  unsigned int v95; // [rsp+1Ch] [rbp-64h]
  char v96; // [rsp+1Ch] [rbp-64h]
  unsigned __int64 v97; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v98; // [rsp+28h] [rbp-58h]
  __int64 v99[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v100; // [rsp+40h] [rbp-40h]

  if ( *(_BYTE *)(a1 + 16) <= 0x10u )
  {
    v12 = a4[1];
    v13 = *(_QWORD *)a1;
    v100 = 257;
    if ( a3 )
    {
      v18 = sub_15A0680(v13, a2, 0);
      if ( *(_BYTE *)(a1 + 16) > 0x10u || *(_BYTE *)(v18 + 16) > 0x10u )
      {
        v15 = (__int64)sub_179D030(v12, (__int64 *)a1, v18, v99, 0, 0);
      }
      else
      {
        v15 = sub_15A2D50((__int64 *)a1, v18, 0, 0, a6, a7, a8);
        v19 = sub_14DBA30(v15, *(_QWORD *)(v12 + 96), 0);
        if ( v19 )
          v15 = v19;
      }
    }
    else
    {
      v14 = sub_15A0680(v13, a2, 0);
      v15 = (__int64)sub_172C310(v12, a1, v14, v99, 0, a6, a7, a8);
    }
    if ( *(_BYTE *)(v15 + 16) <= 0x10u )
    {
      v16 = sub_14DBA30(v15, a5, a4[331]);
      if ( v16 )
        return v16;
    }
    return v15;
  }
  sub_170B990(*a4, a1);
  v21 = a2;
  v22 = *(unsigned __int8 *)(a1 + 16);
  v23 = v22 - 24;
  if ( v22 == 77 )
  {
    if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) != 0 )
    {
      v68 = 0;
      v90 = 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      do
      {
        if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
          v69 = *(_QWORD *)(a1 - 8);
        else
          v69 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
        v94 = v21;
        v70 = sub_179DF30(*(_QWORD *)(v69 + v68), v21, a3, a4, a5);
        v21 = v94;
        v71 = v70;
        if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
          v72 = *(_QWORD *)(a1 - 8);
        else
          v72 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
        v73 = (__int64 *)(v68 + v72);
        if ( *v73 )
        {
          v74 = v73[1];
          v75 = v73[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v75 = v74;
          if ( v74 )
            *(_QWORD *)(v74 + 16) = *(_QWORD *)(v74 + 16) & 3LL | v75;
        }
        *v73 = v71;
        if ( v71 )
        {
          v76 = *(_QWORD *)(v71 + 8);
          v73[1] = v76;
          if ( v76 )
            *(_QWORD *)(v76 + 16) = (unsigned __int64)(v73 + 1) | *(_QWORD *)(v76 + 16) & 3LL;
          v73[2] = (v71 + 8) | v73[2] & 3;
          *(_QWORD *)(v71 + 8) = v73;
        }
        v68 += 24;
      }
      while ( v90 != v68 );
    }
    return a1;
  }
  if ( v23 > 0x35 )
  {
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v52 = *(_QWORD *)(a1 - 8);
    else
      v52 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    v53 = sub_179DF30(*(_QWORD *)(v52 + 24), a2, a3, a4, a5);
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v54 = *(_QWORD **)(a1 - 8);
    else
      v54 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( v54[3] )
    {
      v55 = v54[4];
      v56 = v54[5] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v56 = v55;
      if ( v55 )
        *(_QWORD *)(v55 + 16) = *(_QWORD *)(v55 + 16) & 3LL | v56;
    }
    v54[3] = v53;
    if ( v53 )
    {
      v57 = *(_QWORD *)(v53 + 8);
      v54[4] = v57;
      if ( v57 )
        *(_QWORD *)(v57 + 16) = (unsigned __int64)(v54 + 4) | *(_QWORD *)(v57 + 16) & 3LL;
      v58 = v54[5];
      v59 = v54 + 3;
      v59[2] = (v53 + 8) | v58 & 3;
      *(_QWORD *)(v53 + 8) = v59;
    }
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v60 = *(_QWORD *)(a1 - 8);
    else
      v60 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    v61 = sub_179DF30(*(_QWORD *)(v60 + 48), a2, a3, a4, a5);
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v62 = *(_QWORD **)(a1 - 8);
    else
      v62 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( v62[6] )
    {
      v63 = v62[7];
      v64 = v62[8] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v64 = v63;
      if ( v63 )
        *(_QWORD *)(v63 + 16) = *(_QWORD *)(v63 + 16) & 3LL | v64;
    }
    v62[6] = v61;
    if ( v61 )
    {
      v65 = *(_QWORD *)(v61 + 8);
      v62[7] = v65;
      if ( v65 )
        *(_QWORD *)(v65 + 16) = (unsigned __int64)(v62 + 7) | *(_QWORD *)(v65 + 16) & 3LL;
      v66 = v62[8];
      v67 = v62 + 6;
      v15 = a1;
      v67[2] = (v61 + 8) | v66 & 3;
      *(_QWORD *)(v61 + 8) = v67;
      return v15;
    }
    return a1;
  }
  if ( v23 > 0x18 )
  {
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v38 = *(_QWORD **)(a1 - 8);
    else
      v38 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    v39 = sub_179DF30(*v38, a2, a3, a4, a5);
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v40 = *(__int64 **)(a1 - 8);
    else
      v40 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( *v40 )
    {
      v41 = v40[1];
      v42 = v40[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v42 = v41;
      if ( v41 )
        *(_QWORD *)(v41 + 16) = *(_QWORD *)(v41 + 16) & 3LL | v42;
    }
    *v40 = v39;
    if ( v39 )
    {
      v43 = *(_QWORD *)(v39 + 8);
      v40[1] = v43;
      if ( v43 )
        *(_QWORD *)(v43 + 16) = (unsigned __int64)(v40 + 1) | *(_QWORD *)(v43 + 16) & 3LL;
      v40[2] = (v39 + 8) | v40[2] & 3;
      *(_QWORD *)(v39 + 8) = v40;
    }
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v44 = *(_QWORD *)(a1 - 8);
    else
      v44 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    v45 = sub_179DF30(*(_QWORD *)(v44 + 24), a2, a3, a4, a5);
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v46 = *(_QWORD **)(a1 - 8);
    else
      v46 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( v46[3] )
    {
      v47 = v46[4];
      v48 = v46[5] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v48 = v47;
      if ( v47 )
        *(_QWORD *)(v47 + 16) = *(_QWORD *)(v47 + 16) & 3LL | v48;
    }
    v46[3] = v45;
    if ( v45 )
    {
      v49 = *(_QWORD *)(v45 + 8);
      v46[4] = v49;
      if ( v49 )
        *(_QWORD *)(v49 + 16) = (unsigned __int64)(v46 + 4) | *(_QWORD *)(v49 + 16) & 3LL;
      v50 = v46[5];
      v51 = v46 + 3;
      v15 = a1;
      v51[2] = (v45 + 8) | v50 & 3;
      *(_QWORD *)(v45 + 8) = v51;
      return v15;
    }
    return a1;
  }
  v24 = *(__int64 ***)a1;
  v88 = a4[1];
  v25 = sub_16431D0(*(_QWORD *)a1);
  v28 = *(_BYTE **)(a1 - 24);
  v29 = a2;
  v30 = v25;
  v31 = v28[16];
  if ( v31 == 13 )
  {
    v89 = v28 + 24;
  }
  else
  {
    v26 = *(_QWORD *)v28;
    if ( *(_BYTE *)(*(_QWORD *)v28 + 8LL) == 16 && v31 <= 0x10u )
    {
      v92 = a2;
      v95 = v30;
      v77 = sub_15A1020(v28, a1, v26, v27);
      v30 = v95;
      v29 = v92;
      if ( v77 )
      {
        if ( *(_BYTE *)(v77 + 16) == 13 )
          v89 = (_BYTE *)(v77 + 24);
      }
    }
  }
  v32 = *(_QWORD **)v89;
  if ( *((_DWORD *)v89 + 2) > 0x40u )
    v32 = (_QWORD *)*v32;
  v33 = (unsigned int)v32;
  if ( a3 != ((_BYTE)v22 == 47) )
  {
    if ( v29 == (_DWORD)v32 )
    {
      if ( (_BYTE)v22 == 47 )
      {
        v98 = v30;
        v79 = v30 - v29;
        if ( v30 > 0x40 )
        {
          v91 = v29;
          v96 = v30;
          sub_16A4EF0((__int64)&v97, 0, 0);
          LOBYTE(v29) = v91;
          LOBYTE(v30) = v96;
        }
        else
        {
          v97 = 0;
        }
        if ( v79 )
        {
          if ( v79 > 0x40 )
          {
            sub_16A5260(&v97, 0, v79);
          }
          else
          {
            v80 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v29 + 64 - (unsigned __int8)v30);
            if ( v98 > 0x40 )
              *(_QWORD *)v97 |= v80;
            else
              v97 |= v80;
          }
        }
      }
      else
      {
        v98 = v30;
        v87 = v29 - v30;
        if ( v30 > 0x40 )
        {
          sub_16A4EF0((__int64)&v97, 0, 0);
          v30 = v98;
          v29 = v98 + v87;
        }
        else
        {
          v97 = 0;
        }
        if ( v29 != v30 )
        {
          if ( v29 > 0x3F || v30 > 0x40 )
            sub_16A5260(&v97, v29, v30);
          else
            v97 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v87 + 64) << v29;
        }
      }
      v100 = 257;
      v81 = sub_15A1070((__int64)v24, (__int64)&v97);
      v82 = sub_1729500(v88, *(unsigned __int8 **)(a1 - 48), v81, v99, a6, a7, a8);
      v15 = (__int64)v82;
      if ( v82[16] > 0x17u )
      {
        sub_15F22F0(v82, a1);
        sub_164B7C0(v15, a1);
      }
      if ( v98 > 0x40 && v97 )
        j_j___libc_free_0_0(v97);
      return v15;
    }
    v34 = sub_15A0680((__int64)v24, (unsigned int)v32 - v29, 0);
    if ( *(_QWORD *)(a1 - 24) )
    {
      v35 = *(_QWORD *)(a1 - 16);
      v36 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v36 = v35;
      if ( v35 )
        *(_QWORD *)(v35 + 16) = *(_QWORD *)(v35 + 16) & 3LL | v36;
    }
    *(_QWORD *)(a1 - 24) = v34;
    if ( v34 )
    {
      v37 = *(_QWORD *)(v34 + 8);
      *(_QWORD *)(a1 - 16) = v37;
      if ( v37 )
        *(_QWORD *)(v37 + 16) = (a1 - 16) | *(_QWORD *)(v37 + 16) & 3LL;
      *(_QWORD *)(a1 - 8) = (v34 + 8) | *(_QWORD *)(a1 - 8) & 3LL;
      *(_QWORD *)(v34 + 8) = a1 - 24;
    }
    goto LABEL_29;
  }
  v78 = v29 + (_DWORD)v32;
  if ( v30 > v78 )
  {
    v83 = sub_15A0680((__int64)v24, v78, 0);
    if ( *(_QWORD *)(a1 - 24) )
    {
      v84 = *(_QWORD *)(a1 - 16);
      v85 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v85 = v84;
      if ( v84 )
        *(_QWORD *)(v84 + 16) = *(_QWORD *)(v84 + 16) & 3LL | v85;
    }
    *(_QWORD *)(a1 - 24) = v83;
    if ( v83 )
    {
      v86 = *(_QWORD *)(v83 + 8);
      *(_QWORD *)(a1 - 16) = v86;
      if ( v86 )
        *(_QWORD *)(v86 + 16) = (a1 - 16) | *(_QWORD *)(v86 + 16) & 3LL;
      *(_QWORD *)(a1 - 8) = (v83 + 8) | *(_QWORD *)(a1 - 8) & 3LL;
      *(_QWORD *)(v83 + 8) = a1 - 24;
    }
LABEL_29:
    if ( (_BYTE)v22 == 47 )
    {
      sub_15F2310(a1, 0);
      v15 = a1;
      sub_15F2330(a1, 0);
      return v15;
    }
    sub_15F2350(a1, 0);
    return a1;
  }
  return sub_15A06D0(v24, v33, v26, v27);
}
