// Function: sub_176E680
// Address: 0x176e680
//
__int64 __fastcall sub_176E680(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // ebx
  bool v7; // al
  unsigned int v8; // ebx
  int v9; // r15d
  unsigned int v10; // r15d
  _QWORD *v11; // r12
  unsigned int v13; // r15d
  char v14; // al
  __int64 v15; // r14
  __int64 v16; // rax
  char v17; // al
  __int64 v18; // r10
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rax
  char v22; // al
  __int64 v23; // r13
  __int64 v24; // r14
  unsigned __int8 *v25; // rax
  __int64 v26; // rdx
  __int64 *v27; // r12
  __int16 v28; // si
  __int64 v29; // rcx
  __int64 v30; // rdi
  unsigned __int8 *v31; // rax
  int v32; // eax
  __int64 v33; // rcx
  char v34; // al
  __int64 v35; // rdx
  __int64 v36; // rcx
  unsigned int v37; // eax
  __int64 v38; // r15
  int v39; // eax
  int v40; // eax
  __int64 ***v41; // rdx
  __int64 v42; // rdi
  int v43; // eax
  int v44; // eax
  __int64 ****v45; // rax
  __int64 v46; // rcx
  __int64 ***v47; // rdx
  __int64 v48; // r14
  __int64 v49; // r12
  __int64 v50; // rax
  unsigned __int8 *v51; // rax
  __int64 v52; // r13
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // rax
  __int64 v56; // r15
  __int64 v57; // rdi
  int v58; // eax
  int v59; // eax
  __int64 ***v60; // rdx
  __int64 v61; // rdi
  int v62; // eax
  int v63; // eax
  __int64 ****v64; // rax
  bool v65; // zf
  __int64 v66; // rbx
  __int64 v67; // r13
  __int16 v68; // r14
  _QWORD **v69; // rax
  _QWORD *v70; // r15
  __int64 *v71; // rax
  __int64 v72; // rsi
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // rax
  __int64 v76; // r13
  _QWORD *v77; // rax
  __int64 ***v78; // [rsp+8h] [rbp-98h]
  __int64 ***v79; // [rsp+8h] [rbp-98h]
  __int64 ***v80; // [rsp+10h] [rbp-90h]
  bool v81; // [rsp+10h] [rbp-90h]
  __int64 v83; // [rsp+18h] [rbp-88h]
  __int64 v84; // [rsp+18h] [rbp-88h]
  __int64 *v85; // [rsp+28h] [rbp-78h] BYREF
  __int64 v86; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v87; // [rsp+38h] [rbp-68h]
  __int64 **v88; // [rsp+40h] [rbp-60h] BYREF
  __int64 v89; // [rsp+48h] [rbp-58h]
  __int16 v90; // [rsp+50h] [rbp-50h] BYREF
  __int64 *v91; // [rsp+58h] [rbp-48h]
  __int64 v92; // [rsp+60h] [rbp-40h]

  v6 = *(_DWORD *)(a4 + 8);
  if ( v6 <= 0x40 )
    v7 = *(_QWORD *)a4 == 1;
  else
    v7 = v6 - 1 == (unsigned int)sub_16A57B0(a4);
  v8 = *(unsigned __int16 *)(a2 + 18);
  BYTE1(v8) &= ~0x80u;
  v9 = v8;
  if ( v7 && v8 == 40 )
  {
    v32 = sub_16431D0(*(_QWORD *)a3);
    if ( v32 )
    {
      v85 = 0;
      v89 = (unsigned int)(v32 - 1);
      v92 = v89;
      v34 = *(_BYTE *)(a3 + 16);
      v86 = 0;
      v88 = &v85;
      v91 = &v86;
      if ( v34 == 51 )
      {
        if ( sub_176AA50((__int64)&v88, *(_QWORD *)(a3 - 48), (__int64)&v85, v33)
          && sub_176DE70((__int64)&v90, *(_QWORD *)(a3 - 24), v73, v74) )
        {
LABEL_41:
          if ( v85 == (__int64 *)v86 && v85 )
          {
            v84 = (__int64)v85;
            v75 = sub_15A0680(*v85, 1, 0);
            v90 = 257;
            v76 = v75;
            v77 = sub_1648A60(56, 2u);
            v11 = v77;
            if ( v77 )
              sub_17582E0((__int64)v77, 40, v84, v76, (__int64)&v88);
            return (__int64)v11;
          }
        }
LABEL_42:
        v9 = *(_WORD *)(a2 + 18) & 0x7FFF;
        goto LABEL_5;
      }
      if ( v34 == 5 && *(_WORD *)(a3 + 18) == 27 )
      {
        if ( sub_176ABC0(
               (__int64)&v88,
               *(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)),
               4LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF),
               v33)
          && sub_176E260((__int64)&v88, a3, v35, v36) )
        {
          goto LABEL_41;
        }
        goto LABEL_42;
      }
    }
  }
LABEL_5:
  v10 = v9 - 32;
  if ( v10 > 1 )
    return 0;
  if ( *(_QWORD *)(a3 - 24) != *(_QWORD *)(a2 - 24) )
  {
LABEL_9:
    if ( v10 > 1 )
      return 0;
    v13 = *(_DWORD *)(a4 + 8);
    if ( v13 <= 0x40 )
    {
      if ( *(_QWORD *)a4 )
        return 0;
    }
    else if ( v13 != (unsigned int)sub_16A57B0(a4) )
    {
      return 0;
    }
    v11 = *(_QWORD **)(a3 + 8);
    if ( !v11 )
      return (__int64)v11;
    v11 = (_QWORD *)v11[1];
    if ( v11 )
      return 0;
    v14 = *(_BYTE *)(a3 + 16);
    if ( v14 != 51 )
    {
      if ( v14 != 5 || *(_WORD *)(a3 + 18) != 27 )
        goto LABEL_17;
      v56 = *(_DWORD *)(a3 + 20) & 0xFFFFFFF;
      v57 = *(_QWORD *)(a3 - 24 * v56);
      v58 = *(unsigned __int8 *)(v57 + 16);
      if ( (unsigned __int8)v58 > 0x17u )
      {
        v59 = v58 - 24;
      }
      else
      {
        if ( (_BYTE)v58 != 5 )
          goto LABEL_17;
        v59 = *(unsigned __int16 *)(v57 + 18);
      }
      if ( v59 != 45 )
        goto LABEL_17;
      v60 = *(__int64 ****)sub_13CF970(v57);
      if ( !v60 )
        goto LABEL_17;
      v61 = *(_QWORD *)(a3 + 24 * (1 - v56));
      v62 = *(unsigned __int8 *)(v61 + 16);
      if ( (unsigned __int8)v62 > 0x17u )
      {
        v63 = v62 - 24;
        goto LABEL_72;
      }
      if ( (_BYTE)v62 == 5 )
      {
        v63 = *(unsigned __int16 *)(v61 + 18);
LABEL_72:
        if ( v63 != 45 )
          goto LABEL_17;
        v79 = v60;
        v64 = (__int64 ****)sub_13CF970(v61);
        v47 = v79;
        v80 = *v64;
        if ( !*v64 )
          goto LABEL_17;
LABEL_61:
        v48 = a1;
        v83 = (__int64)v47;
        v90 = 257;
        v49 = *(_QWORD *)(v48 + 8);
        v50 = sub_15A06D0(*v47, 257, (__int64)v47, v46);
        v51 = sub_17203D0(v49, v8, v83, v50, (__int64 *)&v88);
        v52 = *(_QWORD *)(v48 + 8);
        v90 = 257;
        v27 = (__int64 *)v51;
        v55 = sub_15A06D0(*v80, v8, v53, v54);
        v26 = (__int64)v80;
        v28 = v8;
        v29 = v55;
        v30 = v52;
        goto LABEL_33;
      }
LABEL_17:
      v15 = *(_QWORD *)(a3 - 48);
      goto LABEL_18;
    }
    v15 = *(_QWORD *)(a3 - 48);
    v39 = *(unsigned __int8 *)(v15 + 16);
    if ( (unsigned __int8)v39 > 0x17u )
    {
      v40 = v39 - 24;
    }
    else
    {
      if ( (_BYTE)v39 != 5 )
        goto LABEL_18;
      v40 = *(unsigned __int16 *)(v15 + 18);
    }
    if ( v40 != 45 )
      goto LABEL_18;
    v41 = *(__int64 ****)sub_13CF970(*(_QWORD *)(a3 - 48));
    if ( !v41 )
      goto LABEL_18;
    v42 = *(_QWORD *)(a3 - 24);
    v43 = *(unsigned __int8 *)(v42 + 16);
    if ( (unsigned __int8)v43 > 0x17u )
    {
      v44 = v43 - 24;
      goto LABEL_59;
    }
    if ( (_BYTE)v43 == 5 )
    {
      v44 = *(unsigned __int16 *)(v42 + 18);
LABEL_59:
      if ( v44 != 45 )
        goto LABEL_18;
      v78 = v41;
      v45 = (__int64 ****)sub_13CF970(v42);
      v47 = v78;
      v80 = *v45;
      if ( !*v45 )
        goto LABEL_18;
      goto LABEL_61;
    }
LABEL_18:
    v16 = *(_QWORD *)(v15 + 8);
    if ( !v16 || *(_QWORD *)(v16 + 8) )
      return (__int64)v11;
    v17 = *(_BYTE *)(v15 + 16);
    if ( v17 == 52 )
    {
      v18 = *(_QWORD *)(v15 - 48);
      if ( !v18 )
        return (__int64)v11;
      v19 = *(_QWORD *)(v15 - 24);
      if ( !v19 )
        return (__int64)v11;
    }
    else
    {
      if ( v17 != 5 )
        return (__int64)v11;
      if ( *(_WORD *)(v15 + 18) != 28 )
        return (__int64)v11;
      v18 = *(_QWORD *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
      if ( !v18 )
        return (__int64)v11;
      v19 = *(_QWORD *)(v15 + 24 * (1LL - (*(_DWORD *)(v15 + 20) & 0xFFFFFFF)));
      if ( !v19 )
        return (__int64)v11;
    }
    v20 = *(_QWORD *)(a3 - 24);
    v21 = *(_QWORD *)(v20 + 8);
    if ( !v21 || *(_QWORD *)(v21 + 8) )
      return (__int64)v11;
    v22 = *(_BYTE *)(v20 + 16);
    if ( v22 == 52 )
    {
      v23 = *(_QWORD *)(v20 - 48);
      if ( !v23 )
        return (__int64)v11;
      v24 = *(_QWORD *)(v20 - 24);
      if ( !v24 )
        return (__int64)v11;
    }
    else
    {
      if ( v22 != 5 )
        return (__int64)v11;
      if ( *(_WORD *)(v20 + 18) != 28 )
        return (__int64)v11;
      v23 = *(_QWORD *)(v20 - 24LL * (*(_DWORD *)(v20 + 20) & 0xFFFFFFF));
      if ( !v23 )
        return (__int64)v11;
      v24 = *(_QWORD *)(v20 + 24 * (1LL - (*(_DWORD *)(v20 + 20) & 0xFFFFFFF)));
      if ( !v24 )
        return (__int64)v11;
    }
    v90 = 257;
    v25 = sub_17203D0(*(_QWORD *)(a1 + 8), v8, v18, v19, (__int64 *)&v88);
    v26 = v23;
    v27 = (__int64 *)v25;
    v90 = 257;
    v28 = v8;
    v29 = v24;
    v30 = *(_QWORD *)(a1 + 8);
LABEL_33:
    v31 = sub_17203D0(v30, v28, v26, v29, (__int64 *)&v88);
    v90 = 257;
    return sub_15FB440((unsigned int)(v8 != 32) + 26, v27, (__int64)v31, (__int64)&v88, 0);
  }
  v87 = *(_DWORD *)(a4 + 8);
  if ( v87 > 0x40 )
    sub_16A4FD0((__int64)&v86, (const void **)a4);
  else
    v86 = *(_QWORD *)a4;
  sub_16A7490((__int64)&v86, 1);
  v37 = v87;
  v38 = v86;
  v87 = 0;
  LODWORD(v89) = v37;
  v88 = (__int64 **)v86;
  if ( v37 > 0x40 )
  {
    v81 = (unsigned int)sub_16A5940((__int64)&v88) == 1;
    if ( v38 )
    {
      j_j___libc_free_0_0(v38);
      if ( v87 > 0x40 )
      {
        if ( v86 )
          j_j___libc_free_0_0(v86);
      }
    }
    if ( !v81 )
      goto LABEL_50;
  }
  else if ( !v86 || (v86 & (v86 - 1)) != 0 )
  {
LABEL_50:
    v10 = (*(_WORD *)(a2 + 18) & 0x7FFF) - 32;
    goto LABEL_9;
  }
  v65 = v8 == 32;
  v66 = *(_QWORD *)(a3 - 48);
  v90 = 257;
  v67 = *(_QWORD *)(a3 - 24);
  v68 = 3 * v65 + 34;
  v11 = sub_1648A60(56, 2u);
  if ( v11 )
  {
    v69 = *(_QWORD ***)v66;
    if ( *(_BYTE *)(*(_QWORD *)v66 + 8LL) == 16 )
    {
      v70 = v69[4];
      v71 = (__int64 *)sub_1643320(*v69);
      v72 = (__int64)sub_16463B0(v71, (unsigned int)v70);
    }
    else
    {
      v72 = sub_1643320(*v69);
    }
    sub_15FEC10((__int64)v11, v72, 51, v68, v66, v67, (__int64)&v88, 0);
  }
  return (__int64)v11;
}
