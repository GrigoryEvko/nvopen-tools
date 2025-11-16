// Function: sub_175A220
// Address: 0x175a220
//
_QWORD *__fastcall sub_175A220(__int64 a1, __int64 a2, __int64 a3, __int64 a4, double a5, double a6, double a7)
{
  __int64 v7; // rax
  _QWORD *v8; // r15
  __int64 *v9; // r12
  int v11; // r14d
  __int64 v12; // rcx
  unsigned __int8 v13; // al
  __int64 *v14; // r10
  unsigned int v16; // edx
  int v17; // eax
  bool v18; // al
  bool v19; // al
  _QWORD *v20; // rax
  __int64 v21; // rax
  unsigned int v22; // edx
  bool v23; // al
  _QWORD *v24; // rax
  int v25; // eax
  unsigned int v26; // eax
  const void *v27; // r14
  unsigned int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // r10
  bool v31; // r14
  unsigned int v32; // eax
  __int64 v33; // r14
  unsigned int v34; // eax
  __int64 v35; // rax
  unsigned __int8 *v36; // rax
  __int64 v37; // r13
  _QWORD *v38; // rax
  unsigned int v39; // r14d
  __int64 v40; // r15
  __int64 v41; // r10
  char v42; // al
  __int64 v43; // r14
  __int64 v44; // rax
  unsigned __int8 *v45; // rax
  __int64 v46; // r13
  _QWORD *v47; // rax
  unsigned int v48; // eax
  unsigned int v49; // edx
  _QWORD *v50; // rax
  _QWORD *v51; // rax
  int v52; // eax
  __int64 *v53; // [rsp+8h] [rbp-C8h]
  __int64 v54; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v55; // [rsp+18h] [rbp-B8h]
  unsigned int v56; // [rsp+20h] [rbp-B0h]
  unsigned int v57; // [rsp+20h] [rbp-B0h]
  __int64 *v58; // [rsp+20h] [rbp-B0h]
  unsigned int v59; // [rsp+20h] [rbp-B0h]
  __int64 v60; // [rsp+20h] [rbp-B0h]
  char v61; // [rsp+20h] [rbp-B0h]
  char v62; // [rsp+20h] [rbp-B0h]
  unsigned int v63; // [rsp+20h] [rbp-B0h]
  char v64; // [rsp+20h] [rbp-B0h]
  __int64 *v65; // [rsp+28h] [rbp-A8h]
  __int64 v66; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v67; // [rsp+38h] [rbp-98h]
  __int64 v68; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v69; // [rsp+48h] [rbp-88h]
  __int64 v70; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v71; // [rsp+58h] [rbp-78h]
  unsigned __int64 v72; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v73; // [rsp+68h] [rbp-68h]
  __int16 v74; // [rsp+70h] [rbp-60h]
  unsigned __int64 v75; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v76; // [rsp+88h] [rbp-48h]
  __int16 v77; // [rsp+90h] [rbp-40h]

  v7 = *(_QWORD *)(a3 + 8);
  if ( !v7 )
    return 0;
  v8 = *(_QWORD **)(v7 + 8);
  if ( v8 )
    return 0;
  v9 = *(__int64 **)(a3 - 48);
  v11 = *(_WORD *)(a2 + 18) & 0x7FFF;
  v65 = *(__int64 **)(a3 - 24);
  if ( !sub_15F2380(a3) )
    goto LABEL_6;
  if ( v11 == 38 )
  {
    v22 = *(_DWORD *)(a4 + 8);
    if ( v22 <= 0x40 )
    {
      v12 = 64 - v22;
      if ( *(_QWORD *)a4 != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v22) )
        goto LABEL_85;
    }
    else
    {
      v57 = *(_DWORD *)(a4 + 8);
      if ( v57 != (unsigned int)sub_16A58F0(a4) )
      {
        if ( v57 > 0x40 )
        {
          v23 = v57 == (unsigned int)sub_16A57B0(a4);
LABEL_28:
          if ( !v23 )
            goto LABEL_6;
          v77 = 257;
          v24 = sub_1648A60(56, 2u);
          v8 = v24;
          if ( v24 )
            sub_17582E0((__int64)v24, 38, (__int64)v9, (__int64)v65, (__int64)&v75);
          return v8;
        }
LABEL_85:
        v23 = *(_QWORD *)a4 == 0;
        goto LABEL_28;
      }
    }
    v77 = 257;
    v50 = sub_1648A60(56, 2u);
    v8 = v50;
    if ( v50 )
      sub_17582E0((__int64)v50, 39, (__int64)v9, (__int64)v65, (__int64)&v75);
    return v8;
  }
  if ( v11 != 40 )
    goto LABEL_6;
  v16 = *(_DWORD *)(a4 + 8);
  if ( v16 <= 0x40 )
  {
    v18 = *(_QWORD *)a4 == 0;
  }
  else
  {
    v56 = *(_DWORD *)(a4 + 8);
    v17 = sub_16A57B0(a4);
    v16 = v56;
    v18 = v56 == v17;
  }
  if ( !v18 )
  {
    if ( v16 <= 0x40 )
      v19 = *(_QWORD *)a4 == 1;
    else
      v19 = v16 - 1 == (unsigned int)sub_16A57B0(a4);
    if ( v19 )
    {
      v77 = 257;
      v20 = sub_1648A60(56, 2u);
      v8 = v20;
      if ( v20 )
        sub_17582E0((__int64)v20, 41, (__int64)v9, (__int64)v65, (__int64)&v75);
      return v8;
    }
LABEL_6:
    v13 = *((_BYTE *)v9 + 16);
    v14 = v9 + 3;
    if ( v13 != 13 )
    {
      if ( *(_BYTE *)(*v9 + 8) != 16 )
        return v8;
      if ( v13 > 0x10u )
        return v8;
      v21 = sub_15A1020(v9, a2, *v9, v12);
      if ( !v21 || *(_BYTE *)(v21 + 16) != 13 )
        return v8;
      v14 = (__int64 *)(v21 + 24);
    }
    if ( v11 != 36 )
    {
      if ( v11 != 34 )
        return v8;
      v60 = (__int64)v14;
      sub_13A38D0((__int64)&v68, a4);
      sub_16A7490((__int64)&v68, 1);
      v39 = v69;
      v40 = v68;
      v69 = 0;
      v41 = v60;
      v71 = v39;
      v70 = v68;
      if ( v39 > 0x40 )
      {
        v52 = sub_16A5940((__int64)&v70);
        v41 = v60;
        if ( v52 != 1 )
        {
          v42 = 0;
          goto LABEL_68;
        }
      }
      else if ( !v68 || (v68 & (v68 - 1)) != 0 )
      {
        return 0;
      }
      sub_13A38D0((__int64)&v72, v41);
      if ( v73 > 0x40 )
      {
        sub_16A8890((__int64 *)&v72, (__int64 *)a4);
        v49 = v73;
        v73 = 0;
        v76 = v49;
        v63 = v49;
        v75 = v72;
        v55 = v72;
        v42 = sub_1455820((__int64)&v75, (_QWORD *)a4);
        if ( v63 > 0x40 )
        {
          if ( v55 )
          {
            v64 = v42;
            j_j___libc_free_0_0(v55);
            v42 = v64;
            if ( v73 > 0x40 )
            {
              if ( v72 )
              {
                j_j___libc_free_0_0(v72);
                v42 = v64;
              }
            }
          }
        }
      }
      else
      {
        v72 &= *(_QWORD *)a4;
        v76 = v73;
        v75 = v72;
        v73 = 0;
        v42 = sub_1455820((__int64)&v75, (_QWORD *)a4);
      }
      if ( v39 <= 0x40 )
      {
LABEL_70:
        if ( v69 > 0x40 && v68 )
        {
          v62 = v42;
          j_j___libc_free_0_0(v68);
          v42 = v62;
        }
        if ( v42 )
        {
          v43 = *(_QWORD *)(a1 + 8);
          v74 = 257;
          v44 = sub_15A1070(*v65, a4);
          v45 = sub_172AC10(v43, (__int64)v65, v44, (__int64 *)&v72, a5, a6, a7);
          v77 = 257;
          v46 = (__int64)v45;
          v47 = sub_1648A60(56, 2u);
          v8 = v47;
          if ( v47 )
            sub_17582E0((__int64)v47, 33, v46, (__int64)v9, (__int64)&v75);
          return v8;
        }
        return 0;
      }
LABEL_68:
      if ( v40 )
      {
        v61 = v42;
        j_j___libc_free_0_0(v40);
        v42 = v61;
      }
      goto LABEL_70;
    }
    if ( *(_DWORD *)(a4 + 8) > 0x40u )
    {
      v58 = v14;
      v25 = sub_16A5940(a4);
      v14 = v58;
      if ( v25 != 1 )
        return v8;
    }
    else if ( !*(_QWORD *)a4 || (*(_QWORD *)a4 & (*(_QWORD *)a4 - 1LL)) != 0 )
    {
      return v8;
    }
    v53 = v14;
    sub_13A38D0((__int64)&v72, a4);
    sub_16A7800((__int64)&v72, 1u);
    v26 = v73;
    v27 = (const void *)v72;
    v73 = 0;
    v59 = v26;
    v76 = v26;
    v75 = v72;
    sub_13A38D0((__int64)&v66, a4);
    sub_16A7800((__int64)&v66, 1u);
    v28 = v67;
    v67 = 0;
    v69 = v28;
    v68 = v66;
    if ( v28 > 0x40 )
    {
      sub_16A8890(&v68, v53);
      v48 = v69;
      v30 = v68;
      v69 = 0;
      v71 = v48;
      v70 = v68;
      if ( v48 > 0x40 )
      {
        v54 = v68;
        v31 = sub_16A5220((__int64)&v70, (const void **)&v75);
        if ( v54 )
          j_j___libc_free_0_0(v54);
        goto LABEL_39;
      }
    }
    else
    {
      v29 = *v53 & v66;
      v71 = v28;
      v68 = v29;
      v30 = v29;
      v70 = v29;
      v69 = 0;
    }
    v31 = v27 == (const void *)v30;
LABEL_39:
    if ( v69 > 0x40 && v68 )
      j_j___libc_free_0_0(v68);
    if ( v67 > 0x40 && v66 )
      j_j___libc_free_0_0(v66);
    if ( v59 > 0x40 && v75 )
      j_j___libc_free_0_0(v75);
    if ( v73 > 0x40 && v72 )
      j_j___libc_free_0_0(v72);
    if ( v31 )
    {
      v32 = *(_DWORD *)(a4 + 8);
      v33 = *(_QWORD *)(a1 + 8);
      v74 = 257;
      v69 = v32;
      if ( v32 > 0x40 )
        sub_16A4FD0((__int64)&v68, (const void **)a4);
      else
        v68 = *(_QWORD *)a4;
      sub_16A7800((__int64)&v68, 1u);
      v34 = v69;
      v69 = 0;
      v71 = v34;
      v70 = v68;
      v35 = sub_15A1070(*v65, (__int64)&v70);
      v36 = sub_172AC10(v33, (__int64)v65, v35, (__int64 *)&v72, a5, a6, a7);
      v77 = 257;
      v37 = (__int64)v36;
      v38 = sub_1648A60(56, 2u);
      v8 = v38;
      if ( v38 )
        sub_17582E0((__int64)v38, 32, v37, (__int64)v9, (__int64)&v75);
      if ( v71 > 0x40 && v70 )
        j_j___libc_free_0_0(v70);
      if ( v69 > 0x40 && v68 )
        j_j___libc_free_0_0(v68);
    }
    return v8;
  }
  v77 = 257;
  v51 = sub_1648A60(56, 2u);
  v8 = v51;
  if ( v51 )
    sub_17582E0((__int64)v51, 40, (__int64)v9, (__int64)v65, (__int64)&v75);
  return v8;
}
