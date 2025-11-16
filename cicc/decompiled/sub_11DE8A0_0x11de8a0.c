// Function: sub_11DE8A0
// Address: 0x11de8a0
//
__int64 __fastcall sub_11DE8A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  unsigned __int8 *v5; // r13
  unsigned __int8 *v6; // r14
  __int64 v7; // rbx
  unsigned __int64 v8; // rax
  _QWORD *v9; // r9
  char v10; // bl
  char v11; // al
  size_t v12; // r13
  size_t v13; // rbx
  size_t v14; // rdx
  int v15; // eax
  __int64 v16; // rsi
  __int64 result; // rax
  __int64 *v18; // r9
  __int64 v19; // r8
  __int64 v20; // rdx
  __int64 v21; // rcx
  unsigned __int64 v22; // rax
  char v23; // r11
  size_t v24; // r9
  unsigned __int64 v25; // rax
  char v26; // r11
  unsigned __int64 v27; // r9
  unsigned __int64 v28; // rbx
  __int64 v29; // r15
  unsigned int v30; // eax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 **v33; // r12
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // r15
  __int64 v37; // rax
  char v38; // al
  _QWORD *v39; // rax
  __int64 v40; // r9
  unsigned __int64 v41; // r13
  __int64 v42; // rsi
  unsigned int *v43; // r15
  __int64 v44; // r14
  __int64 v45; // rdx
  __int64 v46; // rdi
  __int64 v47; // rdi
  __int64 (__fastcall *v48)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v49; // r14
  _BYTE *v50; // rax
  __int64 *v51; // rbx
  _QWORD **v52; // r15
  unsigned int v53; // eax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 **v56; // r12
  __int64 v57; // rax
  __int64 v58; // rdi
  __int64 v59; // r14
  __int64 v60; // rax
  char v61; // al
  _QWORD *v62; // rax
  unsigned __int64 v63; // r15
  unsigned int *v64; // r13
  __int64 v65; // r14
  __int64 v66; // rdx
  unsigned int v67; // esi
  __int64 v68; // rdi
  __int64 (__fastcall *v69)(__int64, unsigned int, _BYTE *, __int64); // rax
  _QWORD *v70; // rax
  __int64 v71; // rbx
  __int64 v72; // rax
  __int64 v73; // rbx
  __int64 v74; // r12
  __int64 i; // r12
  __int64 v76; // rdx
  unsigned int v77; // esi
  _QWORD *v78; // rax
  unsigned int *v79; // rbx
  __int64 v80; // r12
  __int64 v81; // rdx
  __int64 v82; // [rsp-10h] [rbp-170h]
  unsigned __int64 v83; // [rsp+0h] [rbp-160h]
  unsigned __int64 v84; // [rsp+8h] [rbp-158h]
  size_t v85; // [rsp+10h] [rbp-150h]
  unsigned __int64 v86; // [rsp+10h] [rbp-150h]
  char v87; // [rsp+10h] [rbp-150h]
  char v88; // [rsp+10h] [rbp-150h]
  char v89; // [rsp+10h] [rbp-150h]
  size_t v90; // [rsp+18h] [rbp-148h]
  char v91; // [rsp+18h] [rbp-148h]
  char v92; // [rsp+18h] [rbp-148h]
  _QWORD **v93; // [rsp+18h] [rbp-148h]
  __int64 v94; // [rsp+18h] [rbp-148h]
  unsigned __int64 v95; // [rsp+18h] [rbp-148h]
  __int64 *v96; // [rsp+20h] [rbp-140h]
  __int64 v97; // [rsp+20h] [rbp-140h]
  _QWORD *v98; // [rsp+20h] [rbp-140h]
  _QWORD *v100; // [rsp+28h] [rbp-138h]
  __int64 v101; // [rsp+28h] [rbp-138h]
  void *s1; // [rsp+30h] [rbp-130h] BYREF
  size_t n; // [rsp+38h] [rbp-128h]
  void *s2; // [rsp+40h] [rbp-120h] BYREF
  size_t v105; // [rsp+48h] [rbp-118h]
  char v106[32]; // [rsp+50h] [rbp-110h] BYREF
  __int16 v107; // [rsp+70h] [rbp-F0h]
  _BYTE v108[32]; // [rsp+80h] [rbp-E0h] BYREF
  __int16 v109; // [rsp+A0h] [rbp-C0h]
  _QWORD v110[4]; // [rsp+B0h] [rbp-B0h] BYREF
  char v111; // [rsp+D0h] [rbp-90h]
  char v112; // [rsp+D1h] [rbp-8Fh]
  __m128i v113; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v114; // [rsp+F0h] [rbp-70h]
  __int64 v115; // [rsp+F8h] [rbp-68h]
  __int64 v116; // [rsp+100h] [rbp-60h]
  __int64 v117; // [rsp+108h] [rbp-58h]
  __int64 v118; // [rsp+110h] [rbp-50h]
  __int64 v119; // [rsp+118h] [rbp-48h]
  __int16 v120; // [rsp+120h] [rbp-40h]

  v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v5 = *(unsigned __int8 **)(a2 - 32 * v4);
  v6 = *(unsigned __int8 **)(a2 + 32 * (1 - v4));
  if ( v6 == v5 )
    return sub_AD64C0(*(_QWORD *)(a2 + 8), 0, 0);
  v114 = 0;
  v115 = 0;
  v7 = *(_QWORD *)(a2 + 32 * (2 - v4));
  v8 = *(_QWORD *)(a1 + 16);
  v116 = 0;
  v117 = 0;
  v113 = (__m128i)v8;
  v120 = 257;
  v118 = 0;
  v119 = 0;
  if ( !(unsigned __int8)sub_9B6260(v7, &v113, 0) )
  {
    if ( *(_BYTE *)v7 == 17 )
      goto LABEL_4;
    return sub_11DCF00(a2, (__int64)v5, (__int64)v6, v7, 1, (unsigned int **)a3);
  }
  v113.m128i_i64[0] = 0x100000000LL;
  sub_11DA4B0(a2, v113.m128i_i32, 2);
  if ( *(_BYTE *)v7 != 17 )
    return sub_11DCF00(a2, (__int64)v5, (__int64)v6, v7, 1, (unsigned int **)a3);
LABEL_4:
  v9 = *(_QWORD **)(v7 + 24);
  if ( *(_DWORD *)(v7 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  if ( !v9 )
    return sub_AD64C0(*(_QWORD *)(a2 + 8), 0, 0);
  if ( v9 == (_QWORD *)1 )
  {
    v18 = *(__int64 **)(a1 + 24);
    v19 = *(_QWORD *)(a1 + 16);
    v20 = v7;
    v21 = a3;
    goto LABEL_26;
  }
  v90 = (size_t)v9;
  s1 = 0;
  n = 0;
  s2 = 0;
  v105 = 0;
  v10 = sub_98B0F0((__int64)v5, &s1, 1u);
  v11 = sub_98B0F0((__int64)v6, &s2, 1u);
  if ( v10 )
  {
    if ( v11 )
    {
      v12 = v90;
      v13 = v90;
      if ( n <= v90 )
        v12 = n;
      if ( v105 <= v90 )
        v13 = v105;
      v14 = v13;
      if ( v12 <= v13 )
        v14 = v12;
      if ( v14 && (v15 = memcmp(s1, s2, v14)) != 0 )
      {
        v16 = ((__int64)v15 >> 63) | 1;
      }
      else if ( v12 == v13 )
      {
        v16 = 0;
      }
      else
      {
        v16 = -(__int64)(v12 < v13) | 1;
      }
      return sub_AD64C0(*(_QWORD *)(a2 + 8), v16, 0);
    }
    if ( !n )
    {
      v33 = *(__int64 ***)(a2 + 8);
      v107 = 257;
      v109 = 257;
      v34 = sub_BCB2B0(*(_QWORD **)(a3 + 72));
      v35 = *(_QWORD *)(a3 + 48);
      v112 = 1;
      v36 = v34;
      v111 = 3;
      v110[0] = "strcmpload";
      v37 = sub_AA4E30(v35);
      v38 = sub_AE5020(v37, v36);
      LOWORD(v116) = 257;
      v87 = v38;
      v39 = sub_BD2C40(80, unk_3F10A14);
      v41 = (unsigned __int64)v39;
      if ( v39 )
      {
        sub_B4D190((__int64)v39, v36, (__int64)v6, (__int64)&v113, 0, v87, 0, 0);
        v40 = v82;
      }
      v42 = v41;
      (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a3 + 88) + 16LL))(
        *(_QWORD *)(a3 + 88),
        v41,
        v110,
        *(_QWORD *)(a3 + 56),
        *(_QWORD *)(a3 + 64),
        v40);
      v43 = *(unsigned int **)a3;
      v44 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
      if ( *(_QWORD *)a3 != v44 )
      {
        do
        {
          v45 = *((_QWORD *)v43 + 1);
          v42 = *v43;
          v43 += 4;
          sub_B99FD0(v41, v42, v45);
        }
        while ( (unsigned int *)v44 != v43 );
      }
      v46 = *(_QWORD *)(v41 + 8);
      if ( v33 == (__int64 **)v46 )
      {
        v49 = v41;
        goto LABEL_56;
      }
      v47 = *(_QWORD *)(a3 + 80);
      v48 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v47 + 120LL);
      if ( v48 == sub_920130 )
      {
        if ( *(_BYTE *)v41 > 0x15u )
          goto LABEL_80;
        v42 = v41;
        if ( (unsigned __int8)sub_AC4810(0x27u) )
          v49 = sub_ADAB70(39, v41, v33, 0);
        else
          v49 = sub_AA93C0(0x27u, v41, (__int64)v33);
      }
      else
      {
        v42 = 39;
        v49 = v48(v47, 39u, (_BYTE *)v41, (__int64)v33);
      }
      if ( v49 )
      {
LABEL_55:
        v46 = *(_QWORD *)(v49 + 8);
LABEL_56:
        v50 = (_BYTE *)sub_AD6530(v46, v42);
        return sub_929DE0((unsigned int **)a3, v50, (_BYTE *)v49, (__int64)v108, 0, 0);
      }
LABEL_80:
      LOWORD(v116) = 257;
      v78 = sub_BD2C40(72, unk_3F10A14);
      v49 = (__int64)v78;
      if ( v78 )
        sub_B515B0((__int64)v78, v41, (__int64)v33, (__int64)&v113, 0, 0);
      v42 = v49;
      (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
        *(_QWORD *)(a3 + 88),
        v49,
        v106,
        *(_QWORD *)(a3 + 56),
        *(_QWORD *)(a3 + 64));
      v79 = *(unsigned int **)a3;
      v80 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
      if ( *(_QWORD *)a3 != v80 )
      {
        do
        {
          v81 = *((_QWORD *)v79 + 1);
          v42 = *v79;
          v79 += 4;
          sub_B99FD0(v49, v42, v81);
        }
        while ( (unsigned int *)v80 != v79 );
      }
      goto LABEL_55;
    }
LABEL_33:
    v85 = v90;
    v91 = v11;
    v22 = sub_98B430((__int64)v5, 8u);
    v23 = v91;
    v24 = v85;
    v84 = v22;
    if ( v22 )
    {
      v113.m128i_i32[0] = 0;
      sub_11DA2E0(a2, (unsigned int *)&v113, 1, v22);
      v24 = v85;
      v23 = v91;
    }
    v86 = v24;
    v92 = v23;
    v25 = sub_98B430((__int64)v6, 8u);
    v26 = v92;
    v27 = v86;
    if ( v25 )
    {
      v83 = v86;
      v88 = v92;
      v95 = v25;
      v113.m128i_i32[0] = 1;
      sub_11DA2E0(a2, (unsigned int *)&v113, 1, v25);
      v27 = v83;
      v26 = v88;
      v25 = v95;
    }
    if ( v10 == 1 || !v26 )
    {
      if ( v26 == 1 || !v10 )
        return 0;
      if ( v84 <= v27 )
        v27 = v84;
      v97 = v27;
      if ( !(unsigned __int8)sub_11DA720(a2, v6, v27, *(_QWORD *)(a1 + 16)) )
        return 0;
      v51 = *(__int64 **)(a1 + 24);
      v94 = *(_QWORD *)(a1 + 16);
      v52 = (_QWORD **)sub_B43CA0(a2);
      v53 = sub_97FA80(*v51, (__int64)v52);
      v54 = sub_BCCE00(*v52, v53);
      v55 = sub_ACD640(v54, v97, 0);
      v19 = v94;
      v21 = a3;
      v18 = v51;
      v20 = v55;
    }
    else
    {
      v28 = v27;
      if ( v25 <= v27 )
        v28 = v25;
      if ( !(unsigned __int8)sub_11DA720(a2, v5, v28, *(_QWORD *)(a1 + 16)) )
        return 0;
      v29 = *(_QWORD *)(a1 + 16);
      v96 = *(__int64 **)(a1 + 24);
      v93 = (_QWORD **)sub_B43CA0(a2);
      v30 = sub_97FA80(*v96, (__int64)v93);
      v31 = sub_BCCE00(*v93, v30);
      v32 = sub_ACD640(v31, v28, 0);
      v18 = v96;
      v21 = a3;
      v19 = v29;
      v20 = v32;
    }
LABEL_26:
    result = sub_11CA900((__int64)v5, (__int64)v6, v20, v21, v19, v18);
    if ( result )
    {
      if ( *(_BYTE *)result == 85 )
        *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
      return result;
    }
    return 0;
  }
  if ( !v11 || v105 )
    goto LABEL_33;
  v56 = *(__int64 ***)(a2 + 8);
  v109 = 257;
  v57 = sub_BCB2B0(*(_QWORD **)(a3 + 72));
  v58 = *(_QWORD *)(a3 + 48);
  v112 = 1;
  v59 = v57;
  v111 = 3;
  v110[0] = "strcmpload";
  v60 = sub_AA4E30(v58);
  v61 = sub_AE5020(v60, v59);
  LOWORD(v116) = 257;
  v89 = v61;
  v62 = sub_BD2C40(80, unk_3F10A14);
  v63 = (unsigned __int64)v62;
  if ( v62 )
    sub_B4D190((__int64)v62, v59, (__int64)v5, (__int64)&v113, 0, v89, 0, 0);
  (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
    *(_QWORD *)(a3 + 88),
    v63,
    v110,
    *(_QWORD *)(a3 + 56),
    *(_QWORD *)(a3 + 64));
  v64 = *(unsigned int **)a3;
  v65 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
  if ( *(_QWORD *)a3 != v65 )
  {
    do
    {
      v66 = *((_QWORD *)v64 + 1);
      v67 = *v64;
      v64 += 4;
      sub_B99FD0(v63, v67, v66);
    }
    while ( (unsigned int *)v65 != v64 );
  }
  if ( v56 == *(__int64 ***)(v63 + 8) )
    return v63;
  v68 = *(_QWORD *)(a3 + 80);
  v69 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v68 + 120LL);
  if ( v69 != sub_920130 )
  {
    result = v69(v68, 39u, (_BYTE *)v63, (__int64)v56);
    goto LABEL_72;
  }
  if ( *(_BYTE *)v63 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x27u) )
      result = sub_ADAB70(39, v63, v56, 0);
    else
      result = sub_AA93C0(0x27u, v63, (__int64)v56);
LABEL_72:
    if ( result )
      return result;
  }
  LOWORD(v116) = 257;
  v70 = sub_BD2C40(72, unk_3F10A14);
  if ( v70 )
  {
    v98 = v70;
    sub_B515B0((__int64)v70, v63, (__int64)v56, (__int64)&v113, 0, 0);
    v70 = v98;
  }
  v71 = a3;
  v100 = v70;
  (*(void (__fastcall **)(_QWORD, _QWORD *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v71 + 88) + 16LL))(
    *(_QWORD *)(v71 + 88),
    v70,
    v108,
    *(_QWORD *)(v71 + 56),
    *(_QWORD *)(v71 + 64));
  v72 = v71;
  v73 = *(_QWORD *)v71;
  v74 = *(unsigned int *)(v72 + 8);
  result = (__int64)v100;
  for ( i = v73 + 16 * v74; i != v73; result = v101 )
  {
    v76 = *(_QWORD *)(v73 + 8);
    v77 = *(_DWORD *)v73;
    v73 += 16;
    v101 = result;
    sub_B99FD0(result, v77, v76);
  }
  return result;
}
