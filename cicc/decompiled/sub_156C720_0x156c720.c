// Function: sub_156C720
// Address: 0x156c720
//
__int64 __fastcall sub_156C720(__int64 *a1, __int64 *a2, char a3)
{
  int v5; // eax
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // r15
  __int64 v12; // r14
  __int64 v13; // r14
  __int64 v14; // r13
  __int64 v15; // r13
  _QWORD *v16; // r13
  int v17; // r15d
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r14
  int v21; // r14d
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rdx
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // r15
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdi
  unsigned __int64 *v38; // r14
  __int64 v39; // rax
  unsigned __int64 v40; // rcx
  __int64 v41; // rsi
  __int64 v42; // rsi
  __int64 v43; // rax
  __int64 v44; // rdi
  __int64 *v45; // r15
  __int64 v46; // rax
  __int64 v47; // rcx
  __int64 v48; // rsi
  __int64 v49; // rsi
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rsi
  __int64 v53; // rax
  __int64 v54; // rsi
  __int64 v55; // rdx
  __int64 v56; // rsi
  __int64 v57; // rax
  __int64 v58; // rdi
  __int64 v59; // rsi
  __int64 v60; // rax
  __int64 v61; // rsi
  __int64 v62; // rdx
  __int64 v63; // rsi
  __int64 v64; // rax
  __int64 v65; // rdi
  __int64 v66; // rsi
  __int64 v67; // rax
  __int64 v68; // rsi
  __int64 v69; // rdx
  __int64 v70; // rsi
  __int64 v71; // rsi
  __int64 v72; // rax
  __int64 v73; // rdi
  __int64 *v74; // r13
  __int64 v75; // rax
  __int64 v76; // rcx
  __int64 v77; // rsi
  __int64 v78; // rsi
  __int64 *v79; // [rsp+10h] [rbp-90h]
  __int64 *v81; // [rsp+18h] [rbp-88h]
  __int64 *v82; // [rsp+18h] [rbp-88h]
  __int64 *v83; // [rsp+18h] [rbp-88h]
  __int64 v84; // [rsp+28h] [rbp-78h] BYREF
  _BYTE v85[16]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v86; // [rsp+40h] [rbp-60h]
  _BYTE v87[16]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v88; // [rsp+60h] [rbp-40h]

  v5 = *((_DWORD *)a2 + 5);
  v6 = *a2;
  v86 = 257;
  v7 = v5 & 0xFFFFFFF;
  v8 = a2[-3 * v7];
  if ( v6 != *(_QWORD *)v8 )
  {
    if ( *(_BYTE *)(v8 + 16) > 0x10u )
    {
      v71 = a2[-3 * v7];
      v88 = 257;
      v72 = sub_15FDBD0(47, v71, v6, v87, 0);
      v73 = a1[1];
      v8 = v72;
      if ( v73 )
      {
        v74 = (__int64 *)a1[2];
        sub_157E9D0(v73 + 40, v72);
        v75 = *(_QWORD *)(v8 + 24);
        v76 = *v74;
        *(_QWORD *)(v8 + 32) = v74;
        v76 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v8 + 24) = v76 | v75 & 7;
        *(_QWORD *)(v76 + 8) = v8 + 24;
        *v74 = *v74 & 7 | (v8 + 24);
      }
      sub_164B780(v8, v85);
      v77 = *a1;
      if ( *a1 )
      {
        v84 = *a1;
        sub_1623A60(&v84, v77, 2);
        if ( *(_QWORD *)(v8 + 48) )
          sub_161E7C0(v8 + 48);
        v78 = v84;
        *(_QWORD *)(v8 + 48) = v84;
        if ( v78 )
          sub_1623210(&v84, v78, v8 + 48);
      }
      v7 = *((_DWORD *)a2 + 5) & 0xFFFFFFF;
    }
    else
    {
      v8 = sub_15A46C0(47, a2[-3 * v7], v6, 0);
      v7 = *((_DWORD *)a2 + 5) & 0xFFFFFFF;
    }
  }
  v86 = 257;
  v9 = 3 * (1 - v7);
  v10 = a2[v9];
  if ( v6 != *(_QWORD *)v10 )
  {
    if ( *(_BYTE *)(v10 + 16) > 0x10u )
    {
      v88 = 257;
      v64 = sub_15FDBD0(47, v10, v6, v87, 0);
      v65 = a1[1];
      v10 = v64;
      if ( v65 )
      {
        v79 = (__int64 *)a1[2];
        sub_157E9D0(v65 + 40, v64);
        v66 = *v79;
        v67 = *(_QWORD *)(v10 + 24) & 7LL;
        *(_QWORD *)(v10 + 32) = v79;
        v66 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v10 + 24) = v66 | v67;
        *(_QWORD *)(v66 + 8) = v10 + 24;
        *v79 = *v79 & 7 | (v10 + 24);
      }
      sub_164B780(v10, v85);
      v68 = *a1;
      if ( *a1 )
      {
        v84 = *a1;
        sub_1623A60(&v84, v68, 2);
        v69 = v10 + 48;
        if ( *(_QWORD *)(v10 + 48) )
        {
          sub_161E7C0(v10 + 48);
          v69 = v10 + 48;
        }
        v70 = v84;
        *(_QWORD *)(v10 + 48) = v84;
        if ( v70 )
          sub_1623210(&v84, v70, v69);
      }
    }
    else
    {
      v10 = sub_15A46C0(47, a2[v9], v6, 0);
    }
  }
  if ( a3 )
  {
    v11 = sub_15A0680(v6, 32, 0);
    v86 = 257;
    if ( *(_BYTE *)(v8 + 16) > 0x10u || *(_BYTE *)(v11 + 16) > 0x10u )
    {
      v88 = 257;
      v25 = sub_15FB440(23, v8, v11, v87, 0);
      v26 = a1[1];
      v12 = v25;
      if ( v26 )
      {
        v81 = (__int64 *)a1[2];
        sub_157E9D0(v26 + 40, v25);
        v27 = *v81;
        v28 = *(_QWORD *)(v12 + 24) & 7LL;
        *(_QWORD *)(v12 + 32) = v81;
        v27 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v12 + 24) = v27 | v28;
        *(_QWORD *)(v27 + 8) = v12 + 24;
        *v81 = *v81 & 7 | (v12 + 24);
      }
      sub_164B780(v12, v85);
      v29 = *a1;
      if ( *a1 )
      {
        v84 = *a1;
        sub_1623A60(&v84, v29, 2);
        v30 = v12 + 48;
        if ( *(_QWORD *)(v12 + 48) )
        {
          sub_161E7C0(v12 + 48);
          v30 = v12 + 48;
        }
        v31 = v84;
        *(_QWORD *)(v12 + 48) = v84;
        if ( v31 )
          sub_1623210(&v84, v31, v30);
      }
    }
    else
    {
      v12 = sub_15A2D50(v8, v11, 0, 0);
    }
    v86 = 257;
    if ( *(_BYTE *)(v12 + 16) > 0x10u || *(_BYTE *)(v11 + 16) > 0x10u )
    {
      v88 = 257;
      v57 = sub_15FB440(25, v12, v11, v87, 0);
      v58 = a1[1];
      v13 = v57;
      if ( v58 )
      {
        v83 = (__int64 *)a1[2];
        sub_157E9D0(v58 + 40, v57);
        v59 = *v83;
        v60 = *(_QWORD *)(v13 + 24) & 7LL;
        *(_QWORD *)(v13 + 32) = v83;
        v59 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v13 + 24) = v59 | v60;
        *(_QWORD *)(v59 + 8) = v13 + 24;
        *v83 = *v83 & 7 | (v13 + 24);
      }
      sub_164B780(v13, v85);
      v61 = *a1;
      if ( *a1 )
      {
        v84 = *a1;
        sub_1623A60(&v84, v61, 2);
        v62 = v13 + 48;
        if ( *(_QWORD *)(v13 + 48) )
        {
          sub_161E7C0(v13 + 48);
          v62 = v13 + 48;
        }
        v63 = v84;
        *(_QWORD *)(v13 + 48) = v84;
        if ( v63 )
          sub_1623210(&v84, v63, v62);
      }
    }
    else
    {
      v13 = sub_15A2DA0(v12, v11, 0);
    }
    v86 = 257;
    if ( *(_BYTE *)(v10 + 16) > 0x10u || *(_BYTE *)(v11 + 16) > 0x10u )
    {
      v88 = 257;
      v50 = sub_15FB440(23, v10, v11, v87, 0);
      v51 = a1[1];
      v14 = v50;
      if ( v51 )
      {
        v82 = (__int64 *)a1[2];
        sub_157E9D0(v51 + 40, v50);
        v52 = *v82;
        v53 = *(_QWORD *)(v14 + 24) & 7LL;
        *(_QWORD *)(v14 + 32) = v82;
        v52 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v14 + 24) = v52 | v53;
        *(_QWORD *)(v52 + 8) = v14 + 24;
        *v82 = *v82 & 7 | (v14 + 24);
      }
      sub_164B780(v14, v85);
      v54 = *a1;
      if ( *a1 )
      {
        v84 = *a1;
        sub_1623A60(&v84, v54, 2);
        v55 = v14 + 48;
        if ( *(_QWORD *)(v14 + 48) )
        {
          sub_161E7C0(v14 + 48);
          v55 = v14 + 48;
        }
        v56 = v84;
        *(_QWORD *)(v14 + 48) = v84;
        if ( v56 )
          sub_1623210(&v84, v56, v55);
      }
    }
    else
    {
      v14 = sub_15A2D50(v10, v11, 0, 0);
    }
    v86 = 257;
    if ( *(_BYTE *)(v14 + 16) > 0x10u || *(_BYTE *)(v11 + 16) > 0x10u )
    {
      v88 = 257;
      v43 = sub_15FB440(25, v14, v11, v87, 0);
      v44 = a1[1];
      v15 = v43;
      if ( v44 )
      {
        v45 = (__int64 *)a1[2];
        sub_157E9D0(v44 + 40, v43);
        v46 = *(_QWORD *)(v15 + 24);
        v47 = *v45;
        *(_QWORD *)(v15 + 32) = v45;
        v47 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v15 + 24) = v47 | v46 & 7;
        *(_QWORD *)(v47 + 8) = v15 + 24;
        *v45 = *v45 & 7 | (v15 + 24);
      }
      sub_164B780(v15, v85);
      v48 = *a1;
      if ( *a1 )
      {
        v84 = *a1;
        sub_1623A60(&v84, v48, 2);
        if ( *(_QWORD *)(v15 + 48) )
          sub_161E7C0(v15 + 48);
        v49 = v84;
        *(_QWORD *)(v15 + 48) = v84;
        if ( v49 )
          sub_1623210(&v84, v49, v15 + 48);
      }
    }
    else
    {
      v15 = sub_15A2DA0(v14, v11, 0);
    }
    v86 = 257;
    if ( *(_BYTE *)(v13 + 16) > 0x10u )
      goto LABEL_39;
  }
  else
  {
    v32 = sub_15A0680(v6, 0xFFFFFFFFLL, 0);
    v88 = 257;
    v33 = v32;
    v34 = sub_1281C00(a1, v8, v32, (__int64)v87);
    v88 = 257;
    v13 = v34;
    v35 = sub_1281C00(a1, v10, v33, (__int64)v87);
    v86 = 257;
    v15 = v35;
    if ( *(_BYTE *)(v13 + 16) > 0x10u )
      goto LABEL_39;
  }
  if ( *(_BYTE *)(v15 + 16) <= 0x10u )
  {
    v16 = (_QWORD *)sub_15A2C20(v13, v15, 0, 0);
    goto LABEL_23;
  }
LABEL_39:
  v88 = 257;
  v36 = sub_15FB440(15, v13, v15, v87, 0);
  v37 = a1[1];
  v16 = (_QWORD *)v36;
  if ( v37 )
  {
    v38 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v37 + 40, v36);
    v39 = v16[3];
    v40 = *v38;
    v16[4] = v38;
    v40 &= 0xFFFFFFFFFFFFFFF8LL;
    v16[3] = v40 | v39 & 7;
    *(_QWORD *)(v40 + 8) = v16 + 3;
    *v38 = *v38 & 7 | (unsigned __int64)(v16 + 3);
  }
  sub_164B780(v16, v85);
  v41 = *a1;
  if ( *a1 )
  {
    v84 = *a1;
    sub_1623A60(&v84, v41, 2);
    if ( v16[6] )
      sub_161E7C0(v16 + 6);
    v42 = v84;
    v16[6] = v84;
    if ( v42 )
      sub_1623210(&v84, v42, v16 + 6);
  }
LABEL_23:
  v17 = *((_DWORD *)a2 + 5) & 0xFFFFFFF;
  if ( *((char *)a2 + 23) >= 0 )
    goto LABEL_29;
  v18 = sub_1648A40(a2);
  v20 = v18 + v19;
  if ( *((char *)a2 + 23) >= 0 )
  {
    if ( !(unsigned int)(v20 >> 4) )
      goto LABEL_29;
LABEL_86:
    BUG();
  }
  if ( !(unsigned int)((v20 - sub_1648A40(a2)) >> 4) )
    goto LABEL_29;
  if ( *((char *)a2 + 23) >= 0 )
    goto LABEL_86;
  v21 = *(_DWORD *)(sub_1648A40(a2) + 8);
  if ( *((char *)a2 + 23) >= 0 )
    BUG();
  v22 = sub_1648A40(a2);
  v17 += v21 - *(_DWORD *)(v22 + v23 - 4);
LABEL_29:
  if ( v17 == 5 )
    return sub_156BB10(
             a1,
             (_BYTE *)a2[3 * (3LL - (*((_DWORD *)a2 + 5) & 0xFFFFFFF))],
             (__int64)v16,
             a2[3 * (2LL - (*((_DWORD *)a2 + 5) & 0xFFFFFFF))]);
  else
    return (__int64)v16;
}
