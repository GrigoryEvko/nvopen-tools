// Function: sub_32762F0
// Address: 0x32762f0
//
__int64 __fastcall sub_32762F0(__int64 a1, int a2, __int64 a3)
{
  __int64 *v4; // rax
  __int64 v5; // r15
  __int64 v6; // rbx
  __int16 *v7; // rax
  __int64 v8; // rsi
  __int16 v9; // dx
  _QWORD *v11; // rax
  __int64 v12; // r12
  __int64 v13; // r13
  unsigned __int16 *v14; // rdx
  int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // rdx
  unsigned __int16 v21; // dx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rbx
  __int64 v26; // rax
  void *v27; // rax
  int v28; // r9d
  unsigned __int16 v29; // bx
  __int64 v30; // rax
  unsigned __int64 v31; // rbx
  __int64 v32; // rdx
  unsigned __int16 v33; // dx
  __int64 v34; // rax
  unsigned __int64 v35; // rax
  __int64 v36; // rdx
  int v37; // esi
  __int64 v38; // rdx
  bool v39; // al
  __int64 v40; // rcx
  __int64 v41; // r8
  unsigned __int16 v42; // ax
  __int64 v43; // rdx
  __int64 v44; // r8
  bool v45; // al
  __int64 v46; // rcx
  __int64 v47; // r8
  unsigned __int16 v48; // ax
  __int64 v49; // rdx
  __int64 v50; // r8
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // rdx
  unsigned __int16 v55; // bx
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rdx
  unsigned __int64 v62; // rbx
  unsigned __int16 v63; // r15
  __int64 v64; // rax
  unsigned __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // r8
  __int64 v70; // rdx
  __int64 v71; // rdx
  __int128 v72; // [rsp-10h] [rbp-140h]
  __int128 v73; // [rsp-10h] [rbp-140h]
  __int64 v74; // [rsp+8h] [rbp-128h]
  int v75; // [rsp+14h] [rbp-11Ch]
  unsigned int v76; // [rsp+18h] [rbp-118h]
  unsigned int v77; // [rsp+18h] [rbp-118h]
  bool v78; // [rsp+20h] [rbp-110h]
  unsigned int v80; // [rsp+30h] [rbp-100h] BYREF
  __int64 v81; // [rsp+38h] [rbp-F8h]
  unsigned __int16 v82; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v83; // [rsp+48h] [rbp-E8h]
  unsigned __int16 v84; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v85; // [rsp+58h] [rbp-D8h]
  __int16 v86; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v87; // [rsp+68h] [rbp-C8h]
  __int64 v88; // [rsp+70h] [rbp-C0h]
  __int64 v89; // [rsp+78h] [rbp-B8h]
  __int64 v90; // [rsp+80h] [rbp-B0h]
  __int64 v91; // [rsp+88h] [rbp-A8h]
  unsigned __int16 v92; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v93; // [rsp+98h] [rbp-98h]
  unsigned __int16 v94; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v95; // [rsp+A8h] [rbp-88h]
  __int64 v96; // [rsp+B0h] [rbp-80h]
  __int64 v97; // [rsp+B8h] [rbp-78h]
  unsigned __int64 v98; // [rsp+C0h] [rbp-70h]
  __int64 v99; // [rsp+C8h] [rbp-68h]
  unsigned __int16 v100; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v101; // [rsp+D8h] [rbp-58h]
  __int64 v102; // [rsp+E0h] [rbp-50h]
  __int64 v103; // [rsp+E8h] [rbp-48h]
  unsigned __int64 v104; // [rsp+F0h] [rbp-40h] BYREF
  __int64 v105; // [rsp+F8h] [rbp-38h]

  v4 = *(__int64 **)(a1 + 40);
  v5 = *v4;
  v6 = *((unsigned int *)v4 + 2);
  v7 = *(__int16 **)(a1 + 48);
  v8 = *(unsigned int *)(v5 + 24);
  v9 = *v7;
  v81 = *((_QWORD *)v7 + 1);
  LOWORD(v80) = v9;
  if ( (unsigned int)(v8 - 220) > 1 )
    return 0;
  v11 = *(_QWORD **)(v5 + 40);
  v75 = *(_DWORD *)(a1 + 24);
  v12 = *v11;
  v13 = v11[1];
  v14 = (unsigned __int16 *)(*(_QWORD *)(*v11 + 48LL) + 16LL * *((unsigned int *)v11 + 2));
  v15 = *v14;
  v16 = *((_QWORD *)v14 + 1);
  v78 = (_DWORD)v8 == 220;
  v82 = v15;
  v83 = v16;
  if ( (_WORD)v15 )
  {
    v8 = (unsigned int)(v15 - 17);
    if ( (unsigned __int16)(v15 - 17) > 0xD3u )
    {
      v86 = v15;
      v87 = v16;
      goto LABEL_6;
    }
    LOWORD(v15) = word_4456580[v15 - 1];
    v38 = 0;
  }
  else
  {
    v74 = v16;
    if ( !sub_30070B0((__int64)&v82) )
    {
      v87 = v74;
      v86 = 0;
LABEL_11:
      v17 = sub_3007260((__int64)&v86);
      v88 = v17;
      v89 = v20;
      goto LABEL_12;
    }
    LOWORD(v15) = sub_3009970((__int64)&v82, v8, v74, v18, v19);
  }
  v86 = v15;
  v87 = v38;
  if ( !(_WORD)v15 )
    goto LABEL_11;
LABEL_6:
  if ( (_WORD)v15 == 1 || (unsigned __int16)(v15 - 504) <= 7u )
    goto LABEL_82;
  v17 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v15 - 16];
LABEL_12:
  v21 = v80;
  v76 = v17 - v78;
  if ( (_WORD)v80 )
  {
    if ( (unsigned __int16)(v80 - 17) > 0xD3u )
    {
LABEL_14:
      v22 = v81;
      goto LABEL_15;
    }
    v21 = word_4456580[(unsigned __int16)v80 - 1];
    v22 = 0;
  }
  else
  {
    v39 = sub_30070B0((__int64)&v80);
    v21 = 0;
    if ( !v39 )
      goto LABEL_14;
    v42 = sub_3009970((__int64)&v80, v8, 0, v40, v41);
    v44 = v43;
    v21 = v42;
    v22 = v44;
  }
LABEL_15:
  v84 = v21;
  v85 = v22;
  if ( v21 )
  {
    if ( v21 == 1 || (unsigned __int16)(v21 - 504) <= 7u )
      goto LABEL_82;
    v23 = *(_QWORD *)&byte_444C4A0[16 * v21 - 16];
  }
  else
  {
    v23 = sub_3007260((__int64)&v84);
    v90 = v23;
    v91 = v24;
  }
  if ( v76 < (unsigned int)v23 )
    LODWORD(v23) = v76;
  v25 = *(_QWORD *)(v5 + 48) + 16 * v6;
  v77 = v23;
  v26 = *(_QWORD *)(v25 + 8);
  LOWORD(v104) = *(_WORD *)v25;
  v105 = v26;
  v27 = sub_300AC80((unsigned __int16 *)&v104, v8);
  if ( (unsigned int)sub_C336A0((__int64)v27) < v77 )
    return 0;
  v29 = v80;
  if ( (_WORD)v80 )
  {
    if ( (unsigned __int16)(v80 - 17) > 0xD3u )
    {
LABEL_22:
      v30 = v81;
      goto LABEL_23;
    }
    v29 = word_4456580[(unsigned __int16)v80 - 1];
    v30 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v80) )
      goto LABEL_22;
    v29 = sub_3009970((__int64)&v80, v8, v51, v52, v53);
    v30 = v54;
  }
LABEL_23:
  v94 = v29;
  v95 = v30;
  if ( v29 )
  {
    if ( v29 == 1 || (unsigned __int16)(v29 - 504) <= 7u )
      goto LABEL_82;
    v31 = *(_QWORD *)&byte_444C4A0[16 * v29 - 16];
  }
  else
  {
    v96 = sub_3007260((__int64)&v94);
    v31 = v96;
    v97 = v32;
  }
  v33 = v82;
  if ( v82 )
  {
    if ( (unsigned __int16)(v82 - 17) > 0xD3u )
    {
LABEL_27:
      v34 = v83;
      goto LABEL_28;
    }
    v33 = word_4456580[v82 - 1];
    v34 = 0;
  }
  else
  {
    v45 = sub_30070B0((__int64)&v82);
    v33 = 0;
    if ( !v45 )
      goto LABEL_27;
    v48 = sub_3009970((__int64)&v82, v8, 0, v46, v47);
    v50 = v49;
    v33 = v48;
    v34 = v50;
  }
LABEL_28:
  v92 = v33;
  v93 = v34;
  if ( v33 )
  {
    if ( v33 == 1 || (unsigned __int16)(v33 - 504) <= 7u )
      goto LABEL_82;
    v35 = *(_QWORD *)&byte_444C4A0[16 * v33 - 16];
  }
  else
  {
    v35 = sub_3007260((__int64)&v92);
    v98 = v35;
    v99 = v36;
  }
  if ( v35 >= v31 )
  {
    v55 = v80;
    if ( (_WORD)v80 )
    {
      if ( (unsigned __int16)(v80 - 17) > 0xD3u )
        goto LABEL_54;
      v55 = word_4456580[(unsigned __int16)v80 - 1];
      v59 = 0;
    }
    else
    {
      if ( !sub_30070B0((__int64)&v80) )
      {
LABEL_54:
        v59 = v81;
        goto LABEL_55;
      }
      v55 = sub_3009970((__int64)&v80, v8, v56, v57, v58);
      v59 = v71;
    }
LABEL_55:
    LOWORD(v104) = v55;
    v105 = v59;
    if ( v55 )
    {
      if ( v55 == 1 || (unsigned __int16)(v55 - 504) <= 7u )
        goto LABEL_82;
      v62 = *(_QWORD *)&byte_444C4A0[16 * v55 - 16];
    }
    else
    {
      v60 = sub_3007260((__int64)&v104);
      v8 = v61;
      v102 = v60;
      v62 = v60;
      v103 = v61;
    }
    v63 = v82;
    if ( v82 )
    {
      if ( (unsigned __int16)(v82 - 17) <= 0xD3u )
      {
        v63 = word_4456580[v82 - 1];
        v64 = 0;
        goto LABEL_60;
      }
    }
    else if ( sub_30070B0((__int64)&v82) )
    {
      v63 = sub_3009970((__int64)&v82, v8, v67, v68, v69);
      v64 = v70;
      goto LABEL_60;
    }
    v64 = v83;
LABEL_60:
    v100 = v63;
    v101 = v64;
    if ( !v63 )
    {
      v65 = sub_3007260((__int64)&v100);
      v104 = v65;
      v105 = v66;
      goto LABEL_62;
    }
    if ( v63 != 1 && (unsigned __int16)(v63 - 504) > 7u )
    {
      v65 = *(_QWORD *)&byte_444C4A0[16 * v63 - 16];
LABEL_62:
      if ( v65 <= v62 )
        return sub_33FB890(a3, v80, v81, v12, v13);
      *((_QWORD *)&v73 + 1) = v13;
      *(_QWORD *)&v73 = v12;
      return sub_33FAF80(a3, 216, a2, v80, v81, v28, v73);
    }
LABEL_82:
    BUG();
  }
  if ( v75 != 226 || (v37 = 213, !v78) )
    v37 = 214;
  *((_QWORD *)&v72 + 1) = v13;
  *(_QWORD *)&v72 = v12;
  return sub_33FAF80(a3, v37, a2, v80, v81, v28, v72);
}
