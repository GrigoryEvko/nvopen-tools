// Function: sub_11DDEE0
// Address: 0x11ddee0
//
__int64 __fastcall sub_11DDEE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdx
  unsigned __int8 *v6; // r15
  unsigned __int8 *v7; // r13
  char v8; // bl
  char v9; // al
  size_t v10; // r12
  size_t v11; // rbx
  size_t v12; // rdx
  int v13; // eax
  __int64 v14; // rsi
  __int64 result; // rax
  unsigned __int64 v16; // rax
  char v17; // r8
  unsigned __int64 v18; // r9
  _QWORD **v19; // rbx
  unsigned int v20; // eax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 *v23; // r9
  __int64 v24; // r8
  __int64 v25; // rdx
  __int64 **v26; // rax
  _QWORD *v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // r13
  __int64 v31; // rax
  char v32; // al
  _QWORD *v33; // rax
  unsigned __int64 v34; // r14
  unsigned int *v35; // r13
  __int64 v36; // r15
  __int64 v37; // rdx
  unsigned int v38; // esi
  __int64 v39; // rdi
  __int64 (__fastcall *v40)(__int64, unsigned int, _BYTE *, __int64); // rax
  _QWORD *v41; // rax
  __int64 v42; // rdx
  unsigned int *v43; // rbx
  __int64 i; // r12
  __int64 v45; // rdx
  unsigned int v46; // esi
  unsigned __int64 v47; // rax
  __int64 v48; // rax
  unsigned __int64 v49; // r9
  _QWORD **v50; // rbx
  unsigned int v51; // eax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 **v54; // rax
  _QWORD *v55; // rdi
  __int64 v56; // rax
  __int64 v57; // rdi
  __int64 v58; // r15
  __int64 v59; // rax
  char v60; // al
  _QWORD *v61; // rax
  __int64 v62; // r9
  unsigned __int64 v63; // r14
  __int64 v64; // rsi
  unsigned int *v65; // r13
  __int64 v66; // r15
  __int64 v67; // rdx
  __int64 v68; // rdi
  __int64 v69; // rdi
  __int64 (__fastcall *v70)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v71; // r13
  _BYTE *v72; // rax
  _QWORD **v73; // rbx
  unsigned int v74; // eax
  __int64 v75; // rax
  __int64 v76; // rax
  _QWORD *v77; // rax
  unsigned int *v78; // rbx
  __int64 v79; // r14
  __int64 v80; // rdx
  __int64 v81; // [rsp-10h] [rbp-140h]
  __int64 v82; // [rsp+0h] [rbp-130h]
  char v83; // [rsp+0h] [rbp-130h]
  unsigned __int64 v84; // [rsp+8h] [rbp-128h]
  __int64 v85; // [rsp+8h] [rbp-128h]
  char v86; // [rsp+8h] [rbp-128h]
  char v87; // [rsp+8h] [rbp-128h]
  char v88; // [rsp+10h] [rbp-120h]
  __int64 v89; // [rsp+10h] [rbp-120h]
  unsigned __int64 v90; // [rsp+10h] [rbp-120h]
  __int64 v91; // [rsp+10h] [rbp-120h]
  unsigned __int64 v92; // [rsp+10h] [rbp-120h]
  __int64 v93; // [rsp+10h] [rbp-120h]
  __int64 *v94; // [rsp+18h] [rbp-118h]
  __int64 **v95; // [rsp+18h] [rbp-118h]
  _QWORD *v96; // [rsp+18h] [rbp-118h]
  _QWORD *v97; // [rsp+18h] [rbp-118h]
  __int64 v98; // [rsp+18h] [rbp-118h]
  __int64 *v99; // [rsp+18h] [rbp-118h]
  __int64 **v100; // [rsp+18h] [rbp-118h]
  __int64 *v101; // [rsp+18h] [rbp-118h]
  void *s1; // [rsp+20h] [rbp-110h] BYREF
  size_t n; // [rsp+28h] [rbp-108h]
  void *s2; // [rsp+30h] [rbp-100h] BYREF
  size_t v105; // [rsp+38h] [rbp-F8h]
  char v106[32]; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v107; // [rsp+60h] [rbp-D0h]
  _BYTE v108[32]; // [rsp+70h] [rbp-C0h] BYREF
  __int16 v109; // [rsp+90h] [rbp-A0h]
  _QWORD v110[4]; // [rsp+A0h] [rbp-90h] BYREF
  char v111; // [rsp+C0h] [rbp-70h]
  char v112; // [rsp+C1h] [rbp-6Fh]
  unsigned int v113[8]; // [rsp+D0h] [rbp-60h] BYREF
  __int16 v114; // [rsp+F0h] [rbp-40h]

  v5 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v6 = *(unsigned __int8 **)(a2 - 32 * v5);
  v7 = *(unsigned __int8 **)(a2 + 32 * (1 - v5));
  if ( v7 == v6 )
    return sub_AD64C0(*(_QWORD *)(a2 + 8), 0, 0);
  s1 = 0;
  n = 0;
  s2 = 0;
  v105 = 0;
  v8 = sub_98B0F0((__int64)v6, &s1, 1u);
  v9 = sub_98B0F0((__int64)v7, &s2, 1u);
  if ( v8 )
  {
    if ( v9 )
    {
      v10 = n;
      v11 = v105;
      v12 = v105;
      if ( n <= v105 )
        v12 = n;
      if ( v12 && (v13 = memcmp(s1, s2, v12)) != 0 )
      {
        v14 = ((__int64)v13 >> 63) | 1;
      }
      else if ( v10 == v11 )
      {
        v14 = 0;
      }
      else
      {
        v14 = -(__int64)(v10 < v11) | 1;
      }
      return sub_AD64C0(*(_QWORD *)(a2 + 8), v14, 0);
    }
    if ( n )
    {
LABEL_14:
      v88 = v9;
      v84 = sub_98B430((__int64)v6, 8u);
      if ( v84 )
      {
        v113[0] = 0;
        sub_11DA2E0(a2, v113, 1, v84);
        v47 = sub_98B430((__int64)v7, 8u);
        v17 = v88;
        v18 = v47;
        if ( v47 )
        {
          v90 = v47;
          v113[0] = 1;
          sub_11DA2E0(a2, v113, 1, v47);
          v82 = *(_QWORD *)(a1 + 16);
          v99 = *(__int64 **)(a1 + 24);
          v48 = sub_B43CA0(a2);
          v49 = v90;
          v50 = (_QWORD **)v48;
          if ( v84 <= v90 )
            v49 = v84;
          v91 = v49;
          v51 = sub_97FA80(*v99, v48);
          v52 = sub_BCCE00(*v50, v51);
          v53 = sub_ACD640(v52, v91, 0);
          v23 = v99;
          v24 = v82;
          v25 = v53;
          goto LABEL_61;
        }
      }
      else
      {
        v16 = sub_98B430((__int64)v7, 8u);
        v17 = v88;
        v18 = v16;
        if ( v16 )
        {
          v83 = v88;
          v113[0] = 1;
          v92 = v16;
          sub_11DA2E0(a2, v113, 1, v16);
          v18 = v92;
          v17 = v83;
        }
      }
      if ( v8 != 1 && v17 )
      {
        v89 = v18;
        if ( (unsigned __int8)sub_11DA720(a2, v6, v18, *(_QWORD *)(a1 + 16)) )
        {
          v85 = *(_QWORD *)(a1 + 16);
          v94 = *(__int64 **)(a1 + 24);
          v19 = (_QWORD **)sub_B43CA0(a2);
          v20 = sub_97FA80(*v94, (__int64)v19);
          v21 = sub_BCCE00(*v19, v20);
          v22 = sub_ACD640(v21, v89, 0);
          v23 = v94;
          v24 = v85;
          v25 = v22;
          goto LABEL_61;
        }
LABEL_25:
        *(_QWORD *)v113 = 0x100000000LL;
        sub_11DA4B0(a2, (int *)v113, 2);
        return 0;
      }
      if ( v17 == 1 || !v8 || !(unsigned __int8)sub_11DA720(a2, v7, v84, *(_QWORD *)(a1 + 16)) )
        goto LABEL_25;
      v93 = *(_QWORD *)(a1 + 16);
      v101 = *(__int64 **)(a1 + 24);
      v73 = (_QWORD **)sub_B43CA0(a2);
      v74 = sub_97FA80(*v101, (__int64)v73);
      v75 = sub_BCCE00(*v73, v74);
      v76 = sub_ACD640(v75, v84, 0);
      v23 = v101;
      v24 = v93;
      v25 = v76;
LABEL_61:
      result = sub_11CA900((__int64)v6, (__int64)v7, v25, a3, v24, v23);
      if ( result )
      {
        if ( *(_BYTE *)result == 85 )
          *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
        return result;
      }
      return 0;
    }
    v54 = *(__int64 ***)(a2 + 8);
    v55 = *(_QWORD **)(a3 + 72);
    v109 = 257;
    v107 = 257;
    v100 = v54;
    v56 = sub_BCB2B0(v55);
    v57 = *(_QWORD *)(a3 + 48);
    v112 = 1;
    v58 = v56;
    v111 = 3;
    v110[0] = "strcmpload";
    v59 = sub_AA4E30(v57);
    v60 = sub_AE5020(v59, v58);
    v114 = 257;
    v87 = v60;
    v61 = sub_BD2C40(80, unk_3F10A14);
    v63 = (unsigned __int64)v61;
    if ( v61 )
    {
      sub_B4D190((__int64)v61, v58, (__int64)v7, (__int64)v113, 0, v87, 0, 0);
      v62 = v81;
    }
    v64 = v63;
    (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a3 + 88) + 16LL))(
      *(_QWORD *)(a3 + 88),
      v63,
      v110,
      *(_QWORD *)(a3 + 56),
      *(_QWORD *)(a3 + 64),
      v62);
    v65 = *(unsigned int **)a3;
    v66 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
    if ( *(_QWORD *)a3 != v66 )
    {
      do
      {
        v67 = *((_QWORD *)v65 + 1);
        v64 = *v65;
        v65 += 4;
        sub_B99FD0(v63, v64, v67);
      }
      while ( (unsigned int *)v66 != v65 );
    }
    v68 = *(_QWORD *)(v63 + 8);
    if ( v100 == (__int64 **)v68 )
    {
      v71 = v63;
      goto LABEL_59;
    }
    v69 = *(_QWORD *)(a3 + 80);
    v70 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v69 + 120LL);
    if ( v70 == sub_920130 )
    {
      if ( *(_BYTE *)v63 > 0x15u )
        goto LABEL_68;
      v64 = v63;
      if ( (unsigned __int8)sub_AC4810(0x27u) )
        v71 = sub_ADAB70(39, v63, v100, 0);
      else
        v71 = sub_AA93C0(0x27u, v63, (__int64)v100);
    }
    else
    {
      v64 = 39;
      v71 = v70(v69, 39u, (_BYTE *)v63, (__int64)v100);
    }
    if ( v71 )
    {
LABEL_58:
      v68 = *(_QWORD *)(v71 + 8);
LABEL_59:
      v72 = (_BYTE *)sub_AD6530(v68, v64);
      return sub_929DE0((unsigned int **)a3, v72, (_BYTE *)v71, (__int64)v108, 0, 0);
    }
LABEL_68:
    v114 = 257;
    v77 = sub_BD2C40(72, unk_3F10A14);
    v71 = (__int64)v77;
    if ( v77 )
      sub_B515B0((__int64)v77, v63, (__int64)v100, (__int64)v113, 0, 0);
    v64 = v71;
    (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
      *(_QWORD *)(a3 + 88),
      v71,
      v106,
      *(_QWORD *)(a3 + 56),
      *(_QWORD *)(a3 + 64));
    v78 = *(unsigned int **)a3;
    v79 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
    if ( *(_QWORD *)a3 != v79 )
    {
      do
      {
        v80 = *((_QWORD *)v78 + 1);
        v64 = *v78;
        v78 += 4;
        sub_B99FD0(v71, v64, v80);
      }
      while ( (unsigned int *)v79 != v78 );
    }
    goto LABEL_58;
  }
  if ( !v9 || v105 )
    goto LABEL_14;
  v26 = *(__int64 ***)(a2 + 8);
  v27 = *(_QWORD **)(a3 + 72);
  v109 = 257;
  v95 = v26;
  v28 = sub_BCB2B0(v27);
  v29 = *(_QWORD *)(a3 + 48);
  v112 = 1;
  v30 = v28;
  v111 = 3;
  v110[0] = "strcmpload";
  v31 = sub_AA4E30(v29);
  v32 = sub_AE5020(v31, v30);
  v114 = 257;
  v86 = v32;
  v33 = sub_BD2C40(80, unk_3F10A14);
  v34 = (unsigned __int64)v33;
  if ( v33 )
    sub_B4D190((__int64)v33, v30, (__int64)v6, (__int64)v113, 0, v86, 0, 0);
  (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
    *(_QWORD *)(a3 + 88),
    v34,
    v110,
    *(_QWORD *)(a3 + 56),
    *(_QWORD *)(a3 + 64));
  v35 = *(unsigned int **)a3;
  v36 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
  if ( *(_QWORD *)a3 != v36 )
  {
    do
    {
      v37 = *((_QWORD *)v35 + 1);
      v38 = *v35;
      v35 += 4;
      sub_B99FD0(v34, v38, v37);
    }
    while ( (unsigned int *)v36 != v35 );
  }
  if ( v95 == *(__int64 ***)(v34 + 8) )
    return v34;
  v39 = *(_QWORD *)(a3 + 80);
  v40 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v39 + 120LL);
  if ( v40 != sub_920130 )
  {
    result = v40(v39, 39u, (_BYTE *)v34, (__int64)v95);
    goto LABEL_38;
  }
  if ( *(_BYTE *)v34 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x27u) )
      result = sub_ADAB70(39, v34, v95, 0);
    else
      result = sub_AA93C0(0x27u, v34, (__int64)v95);
LABEL_38:
    if ( result )
      return result;
  }
  v114 = 257;
  v41 = sub_BD2C40(72, unk_3F10A14);
  if ( v41 )
  {
    v42 = (__int64)v95;
    v96 = v41;
    sub_B515B0((__int64)v41, v34, v42, (__int64)v113, 0, 0);
    v41 = v96;
  }
  v97 = v41;
  (*(void (__fastcall **)(_QWORD, _QWORD *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
    *(_QWORD *)(a3 + 88),
    v41,
    v108,
    *(_QWORD *)(a3 + 56),
    *(_QWORD *)(a3 + 64));
  v43 = *(unsigned int **)a3;
  result = (__int64)v97;
  for ( i = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8); (unsigned int *)i != v43; result = v98 )
  {
    v45 = *((_QWORD *)v43 + 1);
    v46 = *v43;
    v43 += 4;
    v98 = result;
    sub_B99FD0(result, v46, v45);
  }
  return result;
}
