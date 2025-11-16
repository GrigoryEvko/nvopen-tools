// Function: sub_17508C0
// Address: 0x17508c0
//
__int64 __fastcall sub_17508C0(_QWORD *a1, __int64 *a2, double a3, double a4, double a5)
{
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // r15
  char v9; // al
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rax
  int v15; // eax
  __int64 *v16; // rcx
  __int64 v17; // r14
  __int64 v18; // rax
  unsigned __int8 v19; // al
  _QWORD *v20; // rdx
  unsigned int v21; // eax
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned int v26; // eax
  unsigned int v27; // esi
  char v28; // cl
  __int64 v29; // rdi
  unsigned __int8 *v30; // rax
  __int64 v31; // rcx
  __int64 v32; // r13
  unsigned __int8 *v33; // r15
  __int64 v34; // r14
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdi
  unsigned __int8 *v38; // rax
  __int64 v39; // rdi
  __int64 v40; // r15
  unsigned __int8 *v41; // rax
  __int64 v42; // rdi
  __int64 v43; // r14
  unsigned __int8 *v44; // r13
  __int64 v45; // rax
  __int64 v46; // rdi
  __int64 v47; // rax
  __int64 v48; // rdi
  __int64 *v49; // r15
  __int64 v50; // rax
  bool v51; // al
  __int64 v52; // rax
  __int64 v53; // rdi
  __int64 v54; // rsi
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rsi
  __int64 v58; // rsi
  unsigned __int8 *v59; // rsi
  int v60; // [rsp+4h] [rbp-BCh]
  unsigned int v61; // [rsp+8h] [rbp-B8h]
  __int64 v62; // [rsp+8h] [rbp-B8h]
  __int64 v63; // [rsp+10h] [rbp-B0h]
  __int64 **v64; // [rsp+18h] [rbp-A8h]
  unsigned int v65; // [rsp+20h] [rbp-A0h]
  bool v66; // [rsp+26h] [rbp-9Ah]
  unsigned __int8 v67; // [rsp+27h] [rbp-99h]
  __int64 v68; // [rsp+28h] [rbp-98h]
  __int64 v69; // [rsp+28h] [rbp-98h]
  __int64 *v70; // [rsp+28h] [rbp-98h]
  unsigned __int8 *v71; // [rsp+38h] [rbp-88h] BYREF
  __int64 v72; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v73; // [rsp+48h] [rbp-78h]
  __int64 v74[2]; // [rsp+50h] [rbp-70h] BYREF
  __int16 v75; // [rsp+60h] [rbp-60h]
  __int64 v76; // [rsp+70h] [rbp-50h] BYREF
  __int64 v77; // [rsp+78h] [rbp-48h]
  __int16 v78; // [rsp+80h] [rbp-40h]

  v5 = *(a2 - 3);
  v6 = *(_QWORD *)(v5 + 8);
  if ( !v6 || *(_QWORD *)(v6 + 8) )
    return 0;
  v9 = *(_BYTE *)(v5 + 16);
  if ( v9 == 51 )
  {
    v12 = *(_QWORD *)(v5 - 48);
    if ( !v12 )
      return 0;
    v13 = *(_QWORD *)(v5 - 24);
    if ( !v13 )
      return 0;
  }
  else
  {
    if ( v9 != 5 )
      return 0;
    if ( *(_WORD *)(v5 + 18) != 27 )
      return 0;
    v12 = *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
    if ( !v12 )
      return 0;
    v13 = *(_QWORD *)(v5 + 24 * (1LL - (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)));
    if ( !v13 )
      return 0;
  }
  v14 = *(_QWORD *)(v12 + 8);
  if ( !v14 || *(_QWORD *)(v14 + 8) )
    return 0;
  v15 = *(unsigned __int8 *)(v12 + 16);
  v67 = v15;
  if ( (unsigned __int8)v15 <= 0x17u )
  {
    if ( (_BYTE)v15 != 5 )
      return 0;
    if ( (unsigned int)*(unsigned __int16 *)(v12 + 18) - 23 > 1 )
      return 0;
    v63 = *(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
    if ( !v63 )
      return 0;
    v17 = *(_QWORD *)(v12 + 24 * (1LL - (*(_DWORD *)(v12 + 20) & 0xFFFFFFF)));
    if ( !v17 )
      return 0;
    goto LABEL_25;
  }
  if ( (unsigned int)(v15 - 47) > 1 )
    return 0;
  if ( (*(_BYTE *)(v12 + 23) & 0x40) == 0 )
  {
    v16 = (__int64 *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
    v63 = *v16;
    if ( *v16 )
      goto LABEL_19;
    return 0;
  }
  v16 = *(__int64 **)(v12 - 8);
  v63 = *v16;
  if ( !*v16 )
    return 0;
LABEL_19:
  v17 = v16[3];
  if ( !v17 )
    return 0;
LABEL_25:
  v18 = *(_QWORD *)(v13 + 8);
  if ( !v18 )
    return 0;
  v7 = *(_QWORD *)(v18 + 8);
  if ( v7 )
    return 0;
  v19 = *(_BYTE *)(v13 + 16);
  if ( v19 <= 0x17u )
  {
    if ( v19 != 5 )
      return 0;
    if ( (unsigned int)*(unsigned __int16 *)(v13 + 18) - 23 > 1 )
      return 0;
    if ( *(_QWORD *)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF)) != v63 )
      return 0;
    v68 = *(_QWORD *)(v13 + 24 * (1LL - (*(_DWORD *)(v13 + 20) & 0xFFFFFFF)));
    if ( !v68 )
      return 0;
    v60 = 5;
  }
  else
  {
    v60 = v19;
    if ( (unsigned int)v19 - 47 > 1 )
      return 0;
    v20 = (*(_BYTE *)(v13 + 23) & 0x40) != 0
        ? *(_QWORD **)(v13 - 8)
        : (_QWORD *)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
    if ( *v20 != v63 )
      return 0;
    v68 = v20[3];
    if ( !v68 )
      return 0;
  }
  if ( v67 == v19 )
    return v7;
  v64 = (__int64 **)*a2;
  v21 = sub_16431D0(*a2);
  v23 = v21;
  v65 = v21;
  v76 = v21;
  v77 = v68;
  v24 = *(_QWORD *)(v17 + 8);
  if ( !v24 || *(_QWORD *)(v24 + 8) || (v62 = v23, v51 = sub_1750710(&v76, v17, v23, v22), v23 = v62, !(v66 = v51)) )
  {
    v76 = v23;
    v77 = v17;
    v25 = *(_QWORD *)(v68 + 8);
    if ( !v25 || *(_QWORD *)(v25 + 8) || !sub_1750710(&v76, v68, v23, v22) )
      return v7;
    v68 = v17;
    v66 = 0;
  }
  v26 = sub_16431D0(*(_QWORD *)*(a2 - 3));
  v27 = v65;
  v73 = v26;
  v28 = v65 - v26;
  if ( v26 > 0x40 )
  {
    v61 = v65 - v26;
    sub_16A4EF0((__int64)&v72, 0, 0);
    v26 = v73;
    v28 = v61;
    v27 = v73 + v61;
  }
  else
  {
    v72 = 0;
  }
  if ( v27 != v26 )
  {
    if ( v27 > 0x3F || v26 > 0x40 )
      sub_16A5260(&v72, v27, v26);
    else
      v72 |= 0xFFFFFFFFFFFFFFFFLL >> (v28 + 64) << v27;
  }
  if ( (unsigned __int8)sub_14C1670(v63, (__int64)&v72, a1[333], 0, a1[330], (__int64)a2, a1[332]) )
  {
    v29 = a1[1];
    v78 = 257;
    v30 = sub_1708970(v29, 36, v68, v64, &v76);
    v75 = 257;
    v32 = a1[1];
    v33 = v30;
    if ( v30[16] > 0x10u )
    {
      v78 = 257;
      v52 = sub_15FB530((__int64 *)v30, (__int64)&v76, 0, v31);
      v53 = *(_QWORD *)(v32 + 8);
      v34 = v52;
      if ( v53 )
      {
        v70 = *(__int64 **)(v32 + 16);
        sub_157E9D0(v53 + 40, v52);
        v54 = *v70;
        v55 = *(_QWORD *)(v34 + 24) & 7LL;
        *(_QWORD *)(v34 + 32) = v70;
        v54 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v34 + 24) = v54 | v55;
        *(_QWORD *)(v54 + 8) = v34 + 24;
        *v70 = *v70 & 7 | (v34 + 24);
      }
      sub_164B780(v34, v74);
      v71 = (unsigned __int8 *)v34;
      if ( !*(_QWORD *)(v32 + 80) )
        sub_4263D6(v34, v74, v56);
      (*(void (__fastcall **)(__int64, unsigned __int8 **))(v32 + 88))(v32 + 64, &v71);
      v57 = *(_QWORD *)v32;
      if ( *(_QWORD *)v32 )
      {
        v71 = *(unsigned __int8 **)v32;
        sub_1623A60((__int64)&v71, v57, 2);
        v58 = *(_QWORD *)(v34 + 48);
        if ( v58 )
          sub_161E7C0(v34 + 48, v58);
        v59 = v71;
        *(_QWORD *)(v34 + 48) = v71;
        if ( v59 )
          sub_1623210((__int64)&v71, v59, v34 + 48);
      }
    }
    else
    {
      v34 = sub_15A2B90((__int64 *)v30, 0, 0, v31, a3, a4, a5);
      v35 = sub_14DBA30(v34, *(_QWORD *)(v32 + 96), 0);
      if ( v35 )
        v34 = v35;
    }
    v36 = sub_15A0680((__int64)v64, v65 - 1, 0);
    v37 = a1[1];
    v78 = 257;
    v69 = v36;
    v38 = sub_1729500(v37, v33, v36, &v76, a3, a4, a5);
    v39 = a1[1];
    v40 = (__int64)v38;
    v78 = 257;
    v41 = sub_1729500(v39, (unsigned __int8 *)v34, v69, &v76, a3, a4, a5);
    v42 = a1[1];
    v43 = (__int64)v41;
    v78 = 257;
    v44 = sub_1708970(v42, 36, v63, v64, &v76);
    if ( v66 )
    {
      v45 = v43;
      v43 = v40;
      v40 = v45;
    }
    v46 = a1[1];
    v78 = 257;
    v47 = sub_17066B0(v46, (unsigned int)v67 - 24, (__int64)v44, v40, &v76, 0, a3, a4, a5);
    v48 = a1[1];
    v49 = (__int64 *)v47;
    v78 = 257;
    v50 = sub_17066B0(v48, v60 - 24, (__int64)v44, v43, &v76, 0, a3, a4, a5);
    v78 = 257;
    v7 = sub_15FB440(27, v49, v50, (__int64)&v76, 0);
  }
  if ( v73 > 0x40 && v72 )
    j_j___libc_free_0_0(v72);
  return v7;
}
