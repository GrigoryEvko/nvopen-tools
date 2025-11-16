// Function: sub_1472640
// Address: 0x1472640
//
__int64 __fastcall sub_1472640(__int64 a1, __int64 a2, __int64 a3, _QWORD **a4, unsigned int a5, char a6)
{
  bool v9; // zf
  __int64 v10; // rdi
  unsigned int v11; // ebx
  __int64 v13; // rax
  __int64 v15; // rax
  _QWORD **v16; // r10
  char v17; // r8
  __int64 v18; // r9
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r14
  __int64 v22; // rbx
  bool v23; // al
  __int64 v24; // rbx
  __int64 v25; // rdi
  _QWORD **v26; // r10
  __int64 v27; // r9
  char v28; // r8
  bool v29; // al
  _QWORD **v30; // r10
  bool v31; // al
  __int64 v32; // rax
  __int64 v33; // r14
  unsigned int v34; // ebx
  __int64 v37; // r14
  __int64 v38; // r13
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // r11
  __int64 v46; // rax
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // r14
  __int64 v50; // rbx
  __int64 v51; // rax
  __int64 v52; // rdx
  _QWORD *v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  unsigned int v56; // eax
  bool v57; // sf
  __int64 *v58; // rax
  __int64 *v59; // rsi
  unsigned int v60; // eax
  unsigned int v61; // eax
  unsigned int v62; // edx
  __int64 v63; // rax
  __int64 v64; // r14
  __int64 v65; // r14
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // r14
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rdx
  bool v72; // [rsp+4h] [rbp-FCh]
  __int64 *v73; // [rsp+8h] [rbp-F8h]
  char v74; // [rsp+8h] [rbp-F8h]
  char v75; // [rsp+10h] [rbp-F0h]
  __int64 v76; // [rsp+10h] [rbp-F0h]
  __int64 v77; // [rsp+10h] [rbp-F0h]
  __int64 v78; // [rsp+18h] [rbp-E8h]
  __int64 v79; // [rsp+18h] [rbp-E8h]
  char v80; // [rsp+18h] [rbp-E8h]
  __int64 v82; // [rsp+20h] [rbp-E0h]
  _QWORD **v83; // [rsp+20h] [rbp-E0h]
  _QWORD **v84; // [rsp+20h] [rbp-E0h]
  _QWORD **v85; // [rsp+20h] [rbp-E0h]
  _QWORD **v87; // [rsp+28h] [rbp-D8h]
  _QWORD **v88; // [rsp+28h] [rbp-D8h]
  unsigned int v89; // [rsp+28h] [rbp-D8h]
  __int64 v90; // [rsp+28h] [rbp-D8h]
  __int64 v91; // [rsp+28h] [rbp-D8h]
  __int64 v92; // [rsp+28h] [rbp-D8h]
  __int64 v93; // [rsp+28h] [rbp-D8h]
  __int64 v94; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v95; // [rsp+30h] [rbp-D0h] BYREF
  unsigned int v96; // [rsp+38h] [rbp-C8h]
  __int64 v97; // [rsp+40h] [rbp-C0h] BYREF
  unsigned int v98; // [rsp+48h] [rbp-B8h]
  __int64 v99; // [rsp+50h] [rbp-B0h] BYREF
  unsigned int v100; // [rsp+58h] [rbp-A8h]
  unsigned __int64 v101; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v102; // [rsp+68h] [rbp-98h]
  __int64 v103; // [rsp+70h] [rbp-90h] BYREF
  __int64 v104; // [rsp+80h] [rbp-80h] BYREF
  _BYTE *v105; // [rsp+88h] [rbp-78h]
  _BYTE *v106; // [rsp+90h] [rbp-70h]
  __int64 v107; // [rsp+98h] [rbp-68h]
  int v108; // [rsp+A0h] [rbp-60h]
  _BYTE v109[88]; // [rsp+A8h] [rbp-58h] BYREF

  v9 = *(_WORD *)(a3 + 24) == 0;
  v105 = v109;
  v104 = 0;
  v106 = v109;
  v107 = 4;
  v108 = 0;
  if ( v9 )
  {
    v10 = *(_QWORD *)(a3 + 32);
    v11 = *(_DWORD *)(v10 + 32);
    if ( v11 <= 0x40 )
    {
      if ( !*(_QWORD *)(v10 + 24) )
        goto LABEL_4;
    }
    else if ( v11 == (unsigned int)sub_16A57B0(v10 + 24) )
    {
LABEL_4:
      sub_14573F0(a1, a3);
      goto LABEL_5;
    }
    goto LABEL_9;
  }
  v15 = sub_1457820(a2, a3);
  v16 = a4;
  v17 = a5;
  v18 = v15;
  if ( *(_WORD *)(v15 + 24) != 7 )
  {
    if ( !a6 )
      goto LABEL_9;
    v39 = sub_1493080(a2, a3, a4, &v104, a5, v15);
    v16 = a4;
    v17 = a5;
    v18 = v39;
    if ( !v39 )
      goto LABEL_9;
  }
  if ( v16 != *(_QWORD ***)(v18 + 48) )
    goto LABEL_9;
  v19 = *(_QWORD *)(v18 + 40);
  if ( v19 == 3 )
  {
    v90 = v18;
    v80 = v17;
    v83 = v16;
    v40 = sub_1456040(**(_QWORD **)(v18 + 32));
    v18 = v90;
    if ( *(_BYTE *)(v40 + 8) != 11 )
    {
      v19 = *(_QWORD *)(v90 + 40);
      v17 = v80;
      v16 = v83;
      goto LABEL_13;
    }
    sub_145D730((__int64)&v101, v90, a2);
    if ( (_BYTE)v103 )
    {
      v49 = v101;
      v50 = v102;
      v51 = sub_15A35F0(36, *(_QWORD *)(v101 + 32), *(_QWORD *)(v102 + 32), 0, v47, v48);
      v52 = v51;
      if ( *(_BYTE *)(v51 + 16) == 13 )
      {
        v53 = *(_QWORD **)(v51 + 24);
        if ( *(_DWORD *)(v52 + 32) > 0x40u )
          v53 = (_QWORD *)*v53;
        if ( !v53 )
          v49 = v50;
        v54 = sub_1487810(v90, v49, a2);
        if ( sub_14560B0(v54) )
        {
          sub_14575B0(a1, v49, v49, 0, (__int64)&v104);
          goto LABEL_5;
        }
      }
    }
LABEL_9:
    v13 = sub_1456E90(a2);
    sub_14573F0(a1, v13);
    goto LABEL_5;
  }
LABEL_13:
  if ( v19 != 2 )
    goto LABEL_9;
  v75 = v17;
  v82 = v18;
  v87 = v16;
  v78 = sub_1472270(a2, **(_QWORD **)(v18 + 32), *v16);
  v20 = sub_1472270(a2, *(_QWORD *)(*(_QWORD *)(v82 + 32) + 8LL), *v87);
  v21 = v20;
  if ( *(_WORD *)(v20 + 24) )
    goto LABEL_9;
  v22 = *(_QWORD *)(v20 + 32);
  v73 = (__int64 *)(v22 + 24);
  if ( sub_13D01C0(v22 + 24) )
    goto LABEL_9;
  v23 = sub_13D0200(v73, *(_DWORD *)(v22 + 32) - 1);
  v24 = v78;
  v25 = (__int64)v73;
  v72 = v23;
  v26 = v87;
  v27 = v82;
  v28 = v75;
  if ( !v23 )
  {
    v85 = v87;
    v93 = v27;
    v55 = sub_1480620(a2, v78, 0);
    v28 = v75;
    v26 = v85;
    v24 = v55;
    v27 = v93;
    v25 = *(_QWORD *)(v21 + 32) + 24LL;
  }
  v88 = v26;
  v74 = v28;
  v76 = v27;
  v29 = sub_1455000(v25);
  v30 = v88;
  if ( !v29 )
  {
    v31 = sub_1454FB0(v25);
    v30 = v88;
    if ( !v31 )
    {
      if ( v74 && (*(_BYTE *)(v76 + 26) & 1) != 0 && (unsigned __int8)sub_14691E0(a2, *(_QWORD *)(v76 + 48)) )
      {
        if ( v72 )
          v21 = sub_1480620(a2, v21, 0);
        v68 = sub_1483CF0(a2, v24, v21);
        v69 = sub_1456E90(a2);
        v70 = v68;
        if ( v68 != v69 )
        {
          sub_1477A60(&v101, a2, v68);
          v94 = sub_145CF40(a2, (__int64)&v101);
          sub_135E100((__int64 *)&v101);
          v70 = v94;
        }
        sub_14575B0(a1, v68, v70, 0, (__int64)&v104);
        goto LABEL_5;
      }
      v32 = sub_1480620(a2, v78, 0);
      v33 = *(_QWORD *)(v21 + 32);
      v79 = v32;
      v34 = *(_DWORD *)(v33 + 32);
      v77 = v33 + 24;
      if ( v34 > 0x40 )
      {
        v89 = sub_16A58A0(v77);
      }
      else
      {
        _RDX = *(_QWORD *)(v33 + 24);
        __asm { tzcnt   rax, rdx }
        if ( !_RDX )
          LODWORD(_RAX) = 64;
        if ( v34 < (unsigned int)_RAX )
          LODWORD(_RAX) = *(_DWORD *)(v33 + 32);
        v89 = _RAX;
      }
      if ( (unsigned int)sub_14687F0(a2, v79) < v89 )
      {
        v37 = sub_1456E90(a2);
LABEL_30:
        if ( v37 == sub_1456E90(a2) )
        {
          v38 = v37;
        }
        else
        {
          sub_1477A60(&v101, a2, v37);
          v38 = sub_145CF40(a2, (__int64)&v101);
          sub_135E100((__int64 *)&v101);
        }
        sub_14575B0(a1, v37, v38, 0, (__int64)&v104);
        goto LABEL_5;
      }
      v61 = *(_DWORD *)(v33 + 32);
      LODWORD(v102) = v61;
      if ( v61 > 0x40 )
      {
        sub_16A4FD0(&v101, v77);
        v61 = v102;
        if ( (unsigned int)v102 > 0x40 )
        {
          sub_16A8110(&v101, v89);
          goto LABEL_59;
        }
      }
      else
      {
        v101 = *(_QWORD *)(v33 + 24);
      }
      if ( v89 == v61 )
        v101 = 0;
      else
        v101 >>= v89;
LABEL_59:
      sub_16A5C50(&v95, &v101, v34 + 1);
      v62 = v34 + 1;
      if ( (unsigned int)v102 > 0x40 && v101 )
      {
        j_j___libc_free_0_0(v101);
        v62 = v34 + 1;
      }
      v98 = v62;
      v63 = 1LL << ((unsigned __int8)v34 - (unsigned __int8)v89);
      if ( v62 > 0x40 )
      {
        sub_16A4EF0(&v97, 0, 0);
        v63 = 1LL << ((unsigned __int8)v34 - (unsigned __int8)v89);
        if ( v98 > 0x40 )
        {
          *(_QWORD *)(v97 + 8LL * ((v34 - v89) >> 6)) |= 1LL << ((unsigned __int8)v34 - (unsigned __int8)v89);
          goto LABEL_65;
        }
      }
      else
      {
        v97 = 0;
      }
      v97 |= v63;
LABEL_65:
      sub_16AE1A0(&v101, &v95, &v97);
      sub_16A5A50(&v99, &v101);
      if ( (unsigned int)v102 > 0x40 && v101 )
        j_j___libc_free_0_0(v101);
      LODWORD(v102) = v34;
      v64 = 1LL << v89;
      if ( v34 > 0x40 )
      {
        sub_16A4EF0(&v101, 0, 0);
        if ( (unsigned int)v102 > 0x40 )
        {
          *(_QWORD *)(v101 + 8LL * (v89 >> 6)) |= v64;
LABEL_71:
          v65 = sub_145CF40(a2, (__int64)&v101);
          if ( (unsigned int)v102 > 0x40 && v101 )
            j_j___libc_free_0_0(v101);
          v66 = sub_145CF40(a2, (__int64)&v99);
          v67 = sub_13A5B60(a2, v79, v66, 0, 0);
          v37 = sub_14857A0(a2, v67, v65);
          if ( v100 > 0x40 && v99 )
            j_j___libc_free_0_0(v99);
          if ( v98 > 0x40 && v97 )
            j_j___libc_free_0_0(v97);
          if ( v96 > 0x40 && v95 )
            j_j___libc_free_0_0(v95);
          goto LABEL_30;
        }
      }
      else
      {
        v101 = 0;
      }
      v101 |= v64;
      goto LABEL_71;
    }
  }
  v84 = v30;
  sub_1477A60(&v95, a2, v24);
  v41 = sub_1456040(v24);
  v91 = sub_145CF80(a2, v41, 0, 0);
  v42 = sub_1456040(v24);
  v43 = sub_145CF80(a2, v42, 1, 0);
  v44 = sub_13A5B00(a2, v24, v43, 0, 0);
  v45 = v91;
  v92 = v44;
  if ( (unsigned __int8)sub_148B410(a2, v84, 33, v44, v45) )
  {
    sub_14779E0(&v101, a2, v92);
    sub_158A9F0(&v97, &v101);
    sub_16A7800(&v97, 1);
    v56 = v98;
    v98 = 0;
    v100 = v56;
    v99 = v97;
    v57 = (int)sub_16A9900(&v95, &v99) < 0;
    v58 = &v99;
    if ( v57 )
      v58 = (__int64 *)&v95;
    v59 = v58;
    if ( v96 <= 0x40 && (v60 = *((_DWORD *)v58 + 2), v60 <= 0x40) )
    {
      v71 = *v59;
      v96 = v60;
      v95 = v71 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v60);
    }
    else
    {
      sub_16A51C0(&v95, v59);
    }
    sub_135E100(&v99);
    sub_135E100(&v97);
    sub_135E100(&v103);
    sub_135E100((__int64 *)&v101);
  }
  v46 = sub_145CF40(a2, (__int64)&v95);
  sub_14575B0(a1, v24, v46, 0, (__int64)&v104);
  sub_135E100((__int64 *)&v95);
LABEL_5:
  if ( v106 != v105 )
    _libc_free((unsigned __int64)v106);
  return a1;
}
