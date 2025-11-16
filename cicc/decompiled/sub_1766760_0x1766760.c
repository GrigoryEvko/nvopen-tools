// Function: sub_1766760
// Address: 0x1766760
//
__int64 __fastcall sub_1766760(
        __int64 *a1,
        _WORD *a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  _BYTE *v16; // rdi
  unsigned __int8 v17; // al
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 result; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  unsigned int v25; // r12d
  __int64 v26; // rax
  __int64 v27; // rdi
  char v28; // al
  __int64 v29; // r8
  unsigned __int8 *v30; // r12
  char v31; // al
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // r13
  int v35; // eax
  __int64 v36; // rdx
  __int64 v37; // rdx
  unsigned __int8 v38; // cl
  __int64 *v39; // rax
  __int64 v40; // r13
  __int64 v41; // rbx
  __int64 v42; // rdx
  unsigned __int8 *v43; // rax
  __int64 v44; // rax
  int v45; // eax
  int v46; // eax
  __int64 **v47; // rdx
  int v48; // eax
  unsigned int v49; // esi
  __int64 v50; // rax
  unsigned int v51; // esi
  __int64 v52; // rax
  __int64 v53; // rax
  unsigned int v54; // edx
  const void **v55; // r8
  __int64 v56; // r13
  __int64 v57; // rax
  __int64 v58; // r8
  __int64 v59; // rbx
  const char *v60; // rax
  __int64 v61; // rdx
  unsigned __int8 *v62; // rax
  __int16 v63; // r14
  __int64 v64; // rbx
  __int16 v65; // r14
  const char *v66; // rax
  __int64 v67; // r11
  __int64 v68; // rdx
  const char *v69; // rax
  __int64 v70; // rdx
  unsigned __int8 *v71; // rax
  const void **v72; // [rsp+0h] [rbp-B0h]
  unsigned int v73; // [rsp+8h] [rbp-A8h]
  __int64 v74; // [rsp+8h] [rbp-A8h]
  __int64 *v75; // [rsp+10h] [rbp-A0h]
  __int64 v76; // [rsp+10h] [rbp-A0h]
  __int64 v77; // [rsp+18h] [rbp-98h]
  __int64 v78; // [rsp+18h] [rbp-98h]
  __int64 v79; // [rsp+18h] [rbp-98h]
  __int64 v80; // [rsp+18h] [rbp-98h]
  __int64 v81; // [rsp+18h] [rbp-98h]
  _QWORD v82[2]; // [rsp+20h] [rbp-90h] BYREF
  _QWORD v83[2]; // [rsp+30h] [rbp-80h] BYREF
  const char *v84; // [rsp+40h] [rbp-70h] BYREF
  __int64 v85; // [rsp+48h] [rbp-68h]
  __int16 v86; // [rsp+50h] [rbp-60h]
  __int64 *v87; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v88; // [rsp+68h] [rbp-48h]
  __int16 v89; // [rsp+70h] [rbp-40h]

  v16 = *(_BYTE **)(a3 - 24);
  v17 = v16[16];
  v18 = (__int64)(v16 + 24);
  if ( v17 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v16 + 8LL) != 16 )
      return 0;
    if ( v17 > 0x10u )
      return 0;
    v44 = sub_15A1020(v16, (__int64)a2, *(_QWORD *)v16, a4);
    if ( !v44 || *(_BYTE *)(v44 + 16) != 13 )
      return 0;
    v18 = v44 + 24;
  }
  v19 = *(_QWORD *)(a3 + 8);
  if ( !v19 || *(_QWORD *)(v19 + 8) )
    return 0;
  v20 = *(_QWORD *)(a3 - 48);
  v21 = *(_QWORD *)(v20 + 8);
  if ( !v21 || *(_QWORD *)(v21 + 8) )
  {
LABEL_6:
    result = sub_1766080(a1, (__int64)a2, a3, a4, v18, a5, a6, a7, a8, a9, a10, a11, a12);
    if ( result )
      return result;
    if ( sub_15FF7F0(a2[9] & 0x7FFF) )
      return 0;
    v25 = *(_DWORD *)(a4 + 8);
    if ( v25 <= 0x40 )
    {
      if ( *(_QWORD *)a4 )
        return 0;
    }
    else if ( v25 != (unsigned int)sub_16A57B0(a4) )
    {
      return 0;
    }
    v26 = *(_QWORD *)(*(_QWORD *)(a3 - 48) + 8LL);
    if ( !v26 || *(_QWORD *)(v26 + 8) || !(unsigned __int8)sub_17573B0(*(_BYTE **)(a3 - 24), (__int64)a2, v23, v24) )
      return 0;
    v27 = *(_QWORD *)(a3 - 48);
    v28 = *(_BYTE *)(v27 + 16);
    if ( v28 == 51 )
    {
      v29 = *(_QWORD *)(v27 - 48);
      if ( !v29 )
        return 0;
      v30 = *(unsigned __int8 **)(v27 - 24);
      if ( !v30 )
        return 0;
    }
    else
    {
      if ( v28 != 5 )
        return 0;
      if ( *(_WORD *)(v27 + 18) != 27 )
        return 0;
      v29 = *(_QWORD *)(v27 - 24LL * (*(_DWORD *)(v27 + 20) & 0xFFFFFFF));
      if ( !v29 )
        return 0;
      v30 = *(unsigned __int8 **)(v27 + 24 * (1LL - (*(_DWORD *)(v27 + 20) & 0xFFFFFFF)));
      if ( !v30 )
        return 0;
    }
    v31 = *(_BYTE *)(v29 + 16);
    if ( v31 == 48 )
    {
      if ( *(unsigned __int8 **)(v29 - 48) != v30 )
        return 0;
      v32 = *(_QWORD *)(v29 - 24);
      if ( !v32 )
        return 0;
    }
    else
    {
      if ( v31 != 5 )
        return 0;
      if ( *(_WORD *)(v29 + 18) != 24 )
        return 0;
      if ( *(unsigned __int8 **)(v29 - 24LL * (*(_DWORD *)(v29 + 20) & 0xFFFFFFF)) != v30 )
        return 0;
      v32 = *(_QWORD *)(v29 + 24 * (1LL - (*(_DWORD *)(v29 + 20) & 0xFFFFFFF)));
      if ( !v32 )
        return 0;
    }
    v33 = *(_QWORD *)(a3 + 8);
    v34 = *(_QWORD *)(a3 - 24);
    if ( v33 )
      v35 = *(_QWORD *)(v33 + 8) == 0;
    else
      v35 = 0;
    v36 = *(_QWORD *)(v27 + 8);
    if ( v36 )
      v35 += *(_QWORD *)(v36 + 8) == 0;
    v37 = *(_QWORD *)(v29 + 8);
    v38 = *(_BYTE *)(v32 + 16);
    if ( !v37 || *(_QWORD *)(v37 + 8) )
    {
      if ( v38 <= 0x10u && v35 )
        goto LABEL_30;
    }
    else
    {
      if ( v38 <= 0x10u )
      {
LABEL_30:
        v39 = (__int64 *)sub_15A2D50(*(__int64 **)(a3 - 24), v32, 1u, 0, *(double *)a5.m128_u64, a6, a7);
        v40 = sub_15A2D10(v39, v34, *(double *)a5.m128_u64, a6, a7);
        goto LABEL_31;
      }
      v74 = v29;
      if ( v35 == 2 )
      {
        v76 = a1[1];
        v66 = sub_1649960(v27);
        v67 = a1[1];
        v83[0] = v66;
        v83[1] = v68;
        v81 = v67;
        v89 = 261;
        v87 = v83;
        v69 = sub_1649960(v74);
        v86 = 261;
        v82[0] = v69;
        v82[1] = v70;
        v84 = (const char *)v82;
        v71 = sub_173DC60(v81, v34, v32, (__int64 *)&v84, 1u, 0, *(double *)a5.m128_u64, a6, a7);
        v40 = (__int64)sub_172AC10(v76, (__int64)v71, v34, (__int64 *)&v87, *(double *)a5.m128_u64, a6, a7);
LABEL_31:
        if ( v40 )
        {
          v41 = a1[1];
          v84 = sub_1649960(a3);
          v85 = v42;
          v89 = 261;
          v87 = (__int64 *)&v84;
          v43 = sub_1729500(v41, v30, v40, (__int64 *)&v87, *(double *)a5.m128_u64, a6, a7);
          sub_1593B40((_QWORD *)a2 - 6, (__int64)v43);
          return (__int64)a2;
        }
      }
    }
    return 0;
  }
  v45 = *(unsigned __int8 *)(v20 + 16);
  if ( (unsigned __int8)v45 > 0x17u )
  {
    v46 = v45 - 24;
  }
  else
  {
    if ( (_BYTE)v45 != 5 )
      goto LABEL_6;
    v46 = *(unsigned __int16 *)(v20 + 18);
  }
  if ( v46 != 36 )
    goto LABEL_6;
  v47 = (*(_BYTE *)(v20 + 23) & 0x40) != 0
      ? *(__int64 ***)(v20 - 8)
      : (__int64 **)(v20 - 24LL * (*(_DWORD *)(v20 + 20) & 0xFFFFFFF));
  v75 = *v47;
  if ( !*v47 )
    goto LABEL_6;
  v48 = (unsigned __int16)a2[9];
  BYTE1(v48) &= ~0x80u;
  if ( (unsigned int)(v48 - 32) > 1 )
  {
    v49 = *(_DWORD *)(a4 + 8);
    v50 = *(_QWORD *)a4;
    if ( v49 > 0x40 )
      v50 = *(_QWORD *)(v50 + 8LL * ((v49 - 1) >> 6));
    if ( (v50 & (1LL << ((unsigned __int8)v49 - 1))) != 0 )
      goto LABEL_6;
    v51 = *(_DWORD *)(v18 + 8);
    v52 = *(_QWORD *)v18;
    if ( v51 > 0x40 )
      v52 = *(_QWORD *)(v52 + 8LL * ((v51 - 1) >> 6));
    if ( (v52 & (1LL << ((unsigned __int8)v51 - 1))) != 0 )
      goto LABEL_6;
  }
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
    goto LABEL_6;
  v72 = (const void **)v18;
  v77 = *v75;
  v73 = sub_16431D0(*v75);
  sub_16A5C50((__int64)&v87, (const void **)a4, v73);
  v53 = sub_15A1070(v77, (__int64)&v87);
  v54 = v73;
  v55 = v72;
  v56 = v53;
  if ( v88 > 0x40 && v87 )
  {
    j_j___libc_free_0_0(v87);
    v55 = v72;
    v54 = v73;
  }
  sub_16A5C50((__int64)&v87, v55, v54);
  v57 = sub_15A1070(v77, (__int64)&v87);
  v58 = v57;
  if ( v88 > 0x40 && v87 )
  {
    v78 = v57;
    j_j___libc_free_0_0(v87);
    v58 = v78;
  }
  v59 = a1[1];
  v79 = v58;
  v60 = sub_1649960(a3);
  v89 = 261;
  v84 = v60;
  v85 = v61;
  v87 = (__int64 *)&v84;
  v62 = sub_1729500(v59, (unsigned __int8 *)v75, v79, (__int64 *)&v87, *(double *)a5.m128_u64, a6, a7);
  v63 = a2[9];
  v89 = 257;
  v64 = (__int64)v62;
  v65 = v63 & 0x7FFF;
  result = (__int64)sub_1648A60(56, 2u);
  if ( result )
  {
    v80 = result;
    sub_17582E0(result, v65, v64, v56, (__int64)&v87);
    return v80;
  }
  return result;
}
