// Function: sub_1779430
// Address: 0x1779430
//
__int64 __fastcall sub_1779430(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r13
  _QWORD *v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rbx
  __int64 v10; // r14
  _QWORD *v11; // rax
  __int64 v12; // r15
  __int64 v14; // rbx
  unsigned __int64 v15; // rax
  _QWORD *v16; // rdx
  unsigned __int64 v17; // rcx
  __int64 v18; // rbx
  __int64 i; // r12
  __int64 v20; // r14
  _QWORD *j; // rbx
  char v22; // al
  __int64 v23; // rax
  __int64 v24; // r11
  __int64 v25; // rax
  __int64 v26; // r12
  __int64 v27; // rax
  unsigned __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rsi
  __int64 v31; // r11
  __int64 v32; // rsi
  __int64 v33; // rcx
  int v34; // eax
  __int64 v35; // rdx
  unsigned int v36; // edx
  unsigned int v37; // eax
  unsigned int v38; // edi
  __int64 v39; // rdx
  __int64 *v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rdi
  __int64 v43; // rax
  __int64 v44; // rcx
  int v45; // eax
  __int64 v46; // rcx
  __int64 v47; // rdx
  unsigned int v48; // edx
  unsigned int v49; // eax
  unsigned int v50; // esi
  __int64 v51; // rdx
  __int64 *v52; // rax
  __int64 v53; // rsi
  unsigned __int64 v54; // rdi
  __int64 v55; // rsi
  __int64 v56; // rax
  __int64 v57; // r12
  __int64 *v58; // r12
  __int64 v59; // rax
  unsigned __int64 v60; // rcx
  __int64 *v61; // r13
  __int64 v62; // r15
  unsigned int v63; // ecx
  _QWORD *v64; // rax
  __int64 v65; // r12
  __int64 v66; // rcx
  __int64 v67; // rax
  __int64 v68; // r13
  __int64 v69; // rax
  __int64 v70; // [rsp+8h] [rbp-88h]
  unsigned __int8 v71; // [rsp+14h] [rbp-7Ch]
  unsigned int v72; // [rsp+18h] [rbp-78h]
  __int64 v73; // [rsp+18h] [rbp-78h]
  int v74; // [rsp+20h] [rbp-70h]
  __int64 v75; // [rsp+20h] [rbp-70h]
  __int64 v76; // [rsp+20h] [rbp-70h]
  _QWORD *v77; // [rsp+28h] [rbp-68h]
  __int64 v78; // [rsp+28h] [rbp-68h]
  __int64 v79; // [rsp+28h] [rbp-68h]
  __int64 v80; // [rsp+28h] [rbp-68h]
  char v81; // [rsp+28h] [rbp-68h]
  __int64 v82; // [rsp+28h] [rbp-68h]
  __int64 v83; // [rsp+28h] [rbp-68h]
  const char *v86; // [rsp+40h] [rbp-50h] BYREF
  __int64 v87; // [rsp+48h] [rbp-48h]
  __int64 v88; // [rsp+50h] [rbp-40h]

  v2 = *(_QWORD *)(a2 + 40);
  v3 = sub_157EBA0(v2);
  v4 = sub_15F4DF0(v3, 0);
  v5 = *(_QWORD *)(v4 + 8);
  v6 = v4;
  while ( v5 )
  {
    v7 = sub_1648700(v5);
    v8 = v5;
    v5 = *(_QWORD *)(v5 + 8);
    if ( (unsigned __int8)(*((_BYTE *)v7 + 16) - 25) <= 9u )
      goto LABEL_4;
  }
  v7 = sub_1648700(0);
  v8 = 0;
LABEL_4:
  v9 = v7[5];
  v10 = *(_QWORD *)(v8 + 8);
  if ( v2 == v9 )
    v9 = 0;
  if ( !v10 )
    return 0;
  while ( 1 )
  {
    v11 = sub_1648700(v10);
    if ( (unsigned __int8)(*((_BYTE *)v11 + 16) - 25) <= 9u )
      break;
    v10 = *(_QWORD *)(v10 + 8);
    if ( !v10 )
      return 0;
  }
  v12 = v11[5];
  if ( v12 == v2 )
  {
    v12 = v9;
  }
  else if ( v9 )
  {
    return 0;
  }
  v14 = *(_QWORD *)(v10 + 8);
  if ( v14 )
  {
    while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v14) + 16) - 25) > 9u )
    {
      v14 = *(_QWORD *)(v14 + 8);
      if ( !v14 )
        goto LABEL_17;
    }
    return 0;
  }
LABEL_17:
  if ( v6 == v2 || v12 == v6 )
    return 0;
  v15 = sub_157EBA0(v12);
  if ( !v15 )
    BUG();
  if ( *(_BYTE *)(v15 + 16) != 26 )
    return 0;
  v16 = *(_QWORD **)(v12 + 48);
  v17 = v15 + 24;
  if ( v16 == (_QWORD *)(v15 + 24) )
    return 0;
  if ( (*(_DWORD *)(v15 + 20) & 0xFFFFFFF) != 1 )
  {
    if ( v2 == *(_QWORD *)(v15 - 24) || v2 == *(_QWORD *)(v15 - 48) )
    {
      while ( 1 )
      {
        v18 = v17 - 24;
        if ( *(_BYTE *)(v17 - 8) == 55 )
          break;
        v77 = (_QWORD *)v17;
        if ( (unsigned __int8)sub_15F2ED0(v17 - 24)
          || sub_15F3330(v18)
          || (unsigned __int8)sub_15F3040(v18)
          || v77 == *(_QWORD **)(v12 + 48) )
        {
          return 0;
        }
        v17 = *v77 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v17 )
          BUG();
      }
      if ( *(_QWORD *)(a2 - 24) == *(_QWORD *)(v17 - 48) && sub_15F4220(a2, v17 - 24, 0) )
      {
        for ( i = *(_QWORD *)(v2 + 48); ; i = *(_QWORD *)(i + 8) )
        {
          if ( i )
          {
            v20 = i - 24;
            if ( a2 == i - 24 )
              goto LABEL_57;
          }
          else
          {
            v20 = 0;
          }
          if ( (unsigned __int8)sub_15F2ED0(v20) || sub_15F3330(v20) || (unsigned __int8)sub_15F3040(v20) )
            break;
        }
      }
    }
    return 0;
  }
  for ( j = (_QWORD *)(*(_QWORD *)(v15 + 24) & 0xFFFFFFFFFFFFFFF8LL); ; j = (_QWORD *)(*j & 0xFFFFFFFFFFFFFFF8LL) )
  {
    if ( !j )
      BUG();
    v22 = *((_BYTE *)j - 8);
    if ( v22 != 78 )
      break;
    v23 = *(j - 6);
    if ( *(_BYTE *)(v23 + 16) || (*(_BYTE *)(v23 + 33) & 0x20) == 0 || (unsigned int)(*(_DWORD *)(v23 + 36) - 35) > 3 )
      return 0;
LABEL_40:
    if ( v16 == j )
      return 0;
  }
  if ( v22 == 71 )
  {
    if ( *(_BYTE *)(*(j - 3) + 8LL) != 15 )
      return 0;
    goto LABEL_40;
  }
  if ( v22 != 55 )
    return 0;
  if ( *(_QWORD *)(a2 - 24) != *(j - 6) )
    return 0;
  v18 = (__int64)(j - 3);
  if ( !sub_15F4220(a2, v18, 0) )
    return 0;
LABEL_57:
  v24 = *(_QWORD *)(v18 - 48);
  v25 = *(_QWORD *)(a2 - 48);
  if ( *(_BYTE *)(*(_QWORD *)v25 + 8LL) == 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v24 + 8LL) != 13 )
      goto LABEL_59;
    return 0;
  }
  if ( !v24 )
  {
    v86 = "storemerge";
    LOWORD(v88) = 259;
    BUG();
  }
LABEL_59:
  if ( v25 != v24 )
  {
    v26 = 0;
    v86 = "storemerge";
    LOWORD(v88) = 259;
    v78 = *(_QWORD *)v24;
    v27 = sub_1648B60(64);
    v30 = v78;
    v31 = v27;
    if ( v27 )
    {
      v79 = v27;
      v26 = v27;
      sub_15F1EA0(v27, v30, 53, 0, 0, 0);
      *(_DWORD *)(v79 + 56) = 2;
      sub_164B780(v79, (__int64 *)&v86);
      sub_1648880(v79, *(_DWORD *)(v79 + 56), 1);
      v31 = v79;
    }
    v32 = *(_QWORD *)(a2 + 40);
    v33 = *(_QWORD *)(a2 - 48);
    v34 = *(_DWORD *)(v31 + 20);
    v35 = v34 & 0xFFFFFFF;
    if ( (_DWORD)v35 == *(_DWORD *)(v31 + 56) )
    {
      v73 = *(_QWORD *)(a2 + 40);
      v75 = *(_QWORD *)(a2 - 48);
      v82 = v31;
      sub_15F55D0(v31, v32, v35, v33, v28, v29);
      v31 = v82;
      v32 = v73;
      v33 = v75;
      v34 = *(_DWORD *)(v82 + 20);
    }
    v36 = (v34 + 1) & 0xFFFFFFF;
    v37 = v36 | v34 & 0xF0000000;
    v38 = v36 - 1;
    *(_DWORD *)(v31 + 20) = v37;
    if ( (v37 & 0x40000000) != 0 )
      v39 = *(_QWORD *)(v31 - 8);
    else
      v39 = v26 - 24LL * v36;
    v40 = (__int64 *)(v39 + 24LL * v38);
    if ( *v40 )
    {
      v41 = v40[1];
      v28 = v40[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v28 = v41;
      if ( v41 )
        *(_QWORD *)(v41 + 16) = v28 | *(_QWORD *)(v41 + 16) & 3LL;
    }
    *v40 = v33;
    if ( v33 )
    {
      v42 = *(_QWORD *)(v33 + 8);
      v28 = v33 + 8;
      v40[1] = v42;
      if ( v42 )
      {
        v29 = (__int64)(v40 + 1);
        *(_QWORD *)(v42 + 16) = (unsigned __int64)(v40 + 1) | *(_QWORD *)(v42 + 16) & 3LL;
      }
      v40[2] = v28 | v40[2] & 3;
      *(_QWORD *)(v33 + 8) = v40;
    }
    v43 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
    if ( (*(_BYTE *)(v31 + 23) & 0x40) != 0 )
      v44 = *(_QWORD *)(v31 - 8);
    else
      v44 = v26 - 24 * v43;
    *(_QWORD *)(v44 + 8LL * (unsigned int)(v43 - 1) + 24LL * *(unsigned int *)(v31 + 56) + 8) = v32;
    v45 = *(_DWORD *)(v31 + 20);
    v46 = *(_QWORD *)(v18 - 48);
    v47 = v45 & 0xFFFFFFF;
    if ( (_DWORD)v47 == *(_DWORD *)(v31 + 56) )
    {
      v76 = *(_QWORD *)(v18 - 48);
      v83 = v31;
      sub_15F55D0(v31, v32, v47, v46, v28, v29);
      v31 = v83;
      v46 = v76;
      v45 = *(_DWORD *)(v83 + 20);
    }
    v48 = (v45 + 1) & 0xFFFFFFF;
    v49 = v48 | v45 & 0xF0000000;
    v50 = v48 - 1;
    *(_DWORD *)(v31 + 20) = v49;
    if ( (v49 & 0x40000000) != 0 )
      v51 = *(_QWORD *)(v31 - 8);
    else
      v51 = v26 - 24LL * v48;
    v52 = (__int64 *)(v51 + 24LL * v50);
    if ( *v52 )
    {
      v53 = v52[1];
      v54 = v52[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v54 = v53;
      if ( v53 )
        *(_QWORD *)(v53 + 16) = v54 | *(_QWORD *)(v53 + 16) & 3LL;
    }
    *v52 = v46;
    if ( v46 )
    {
      v55 = *(_QWORD *)(v46 + 8);
      v52[1] = v55;
      if ( v55 )
        *(_QWORD *)(v55 + 16) = (unsigned __int64)(v52 + 1) | *(_QWORD *)(v55 + 16) & 3LL;
      v52[2] = (v46 + 8) | v52[2] & 3;
      *(_QWORD *)(v46 + 8) = v52;
    }
    v56 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
    if ( (*(_BYTE *)(v31 + 23) & 0x40) != 0 )
      v57 = *(_QWORD *)(v31 - 8);
    else
      v57 = v26 - 24 * v56;
    *(_QWORD *)(v57 + 8LL * (unsigned int)(v56 - 1) + 24LL * *(unsigned int *)(v31 + 56) + 8) = v12;
    v58 = *(__int64 **)(v6 + 48);
    if ( !v58 )
      BUG();
    v80 = v31;
    sub_157E9D0(v58[2] + 40, v31);
    v59 = *(_QWORD *)(v80 + 24);
    v60 = *v58 & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v80 + 32) = v58;
    *(_QWORD *)(v80 + 24) = v60 | v59 & 7;
    *(_QWORD *)(v60 + 8) = v80 + 24;
    *v58 = *v58 & 7 | (v80 + 24);
    sub_170B990(*a1, v80);
    v24 = v80;
  }
  v70 = v24;
  v61 = (__int64 *)sub_157EE30(v6);
  v62 = *(_QWORD *)(a2 - 24);
  v63 = *(unsigned __int16 *)(a2 + 18);
  v71 = *(_WORD *)(a2 + 18) & 1;
  v74 = (v63 >> 7) & 7;
  v72 = 1 << (v63 >> 1) >> 1;
  v81 = *(_BYTE *)(a2 + 56);
  v64 = sub_1648A60(64, 2u);
  v65 = (__int64)v64;
  if ( v64 )
    sub_15F9480((__int64)v64, v70, v62, v71, v72, v74, v81, 0);
  if ( !v61 )
    BUG();
  sub_157E9D0(v61[2] + 40, v65);
  v66 = *v61;
  v67 = *(_QWORD *)(v65 + 24);
  *(_QWORD *)(v65 + 32) = v61;
  v66 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(v65 + 24) = v66 | v67 & 7;
  *(_QWORD *)(v66 + 8) = v65 + 24;
  *v61 = *v61 & 7 | (v65 + 24);
  sub_170B990(*a1, v65);
  v68 = sub_15C70A0(v18 + 48);
  v69 = sub_15C70A0(a2 + 48);
  sub_15AC0B0(v65, v69, v68);
  v86 = 0;
  v87 = 0;
  v88 = 0;
  sub_14A8180(a2, (__int64 *)&v86, 0);
  if ( v86 || v87 || v88 )
  {
    sub_14A8180(v18, (__int64 *)&v86, 1);
    sub_1626170(v65, (__int64 *)&v86);
  }
  sub_170BC50((__int64)a1, a2);
  sub_170BC50((__int64)a1, v18);
  return 1;
}
