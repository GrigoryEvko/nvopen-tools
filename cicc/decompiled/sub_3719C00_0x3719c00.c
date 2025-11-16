// Function: sub_3719C00
// Address: 0x3719c00
//
__int64 __fastcall sub_3719C00(__int64 a1, __int64 a2, int a3, char a4)
{
  unsigned int v4; // ebx
  unsigned int v5; // r14d
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rax
  unsigned int v8; // eax
  unsigned int v9; // ebx
  __int64 v10; // r14
  __int64 v11; // r14
  unsigned __int64 v12; // rdx
  unsigned int v13; // eax
  unsigned int v14; // eax
  unsigned __int64 v15; // rax
  unsigned int v16; // r14d
  int v17; // r13d
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rcx
  unsigned int v20; // r14d
  unsigned __int64 v21; // r13
  unsigned int v22; // eax
  unsigned int v23; // edx
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rcx
  unsigned int v26; // r13d
  unsigned int v27; // r14d
  unsigned int v29; // eax
  unsigned int v30; // eax
  bool v31; // zf
  unsigned int v33; // ebx
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rdx
  unsigned int v37; // eax
  unsigned __int64 v38; // rcx
  const void *v39; // rdx
  unsigned int v40; // eax
  unsigned int v41; // r14d
  bool v42; // cc
  unsigned int v43; // eax
  unsigned __int64 v47; // [rsp+40h] [rbp-150h]
  unsigned __int64 v48; // [rsp+50h] [rbp-140h]
  int v49; // [rsp+50h] [rbp-140h]
  unsigned int v50; // [rsp+58h] [rbp-138h]
  unsigned int v51; // [rsp+5Ch] [rbp-134h]
  const void *v52; // [rsp+80h] [rbp-110h] BYREF
  unsigned int v53; // [rsp+88h] [rbp-108h]
  _QWORD *v54; // [rsp+90h] [rbp-100h] BYREF
  unsigned int v55; // [rsp+98h] [rbp-F8h]
  unsigned __int64 v56; // [rsp+A0h] [rbp-F0h] BYREF
  unsigned int v57; // [rsp+A8h] [rbp-E8h]
  unsigned __int64 v58; // [rsp+B0h] [rbp-E0h] BYREF
  unsigned int v59; // [rsp+B8h] [rbp-D8h]
  unsigned __int64 v60; // [rsp+C0h] [rbp-D0h] BYREF
  unsigned int v61; // [rsp+C8h] [rbp-C8h]
  unsigned __int64 v62; // [rsp+D0h] [rbp-C0h] BYREF
  unsigned int v63; // [rsp+D8h] [rbp-B8h]
  unsigned __int64 v64; // [rsp+E0h] [rbp-B0h] BYREF
  unsigned int v65; // [rsp+E8h] [rbp-A8h]
  unsigned __int64 v66; // [rsp+F0h] [rbp-A0h] BYREF
  unsigned int v67; // [rsp+F8h] [rbp-98h]
  unsigned __int64 v68; // [rsp+100h] [rbp-90h] BYREF
  unsigned int v69; // [rsp+108h] [rbp-88h]
  unsigned __int64 v70; // [rsp+110h] [rbp-80h] BYREF
  unsigned int v71; // [rsp+118h] [rbp-78h]
  unsigned __int64 v72; // [rsp+120h] [rbp-70h] BYREF
  unsigned int v73; // [rsp+128h] [rbp-68h]
  unsigned __int64 v74; // [rsp+130h] [rbp-60h] BYREF
  unsigned int v75; // [rsp+138h] [rbp-58h]
  unsigned __int64 v76; // [rsp+140h] [rbp-50h] BYREF
  unsigned int v77; // [rsp+148h] [rbp-48h]
  char v78; // [rsp+150h] [rbp-40h]
  int v79; // [rsp+154h] [rbp-3Ch]

  *(_DWORD *)(a1 + 8) = 1;
  *(_QWORD *)a1 = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v4 = *(_DWORD *)(a2 + 8);
  v5 = v4 - a3;
  v53 = 1;
  v52 = 0;
  v55 = v4;
  if ( v4 <= 0x40 )
  {
    v54 = 0;
    if ( !v5 )
    {
      v57 = v4;
      v8 = v4;
      goto LABEL_7;
    }
    if ( v5 <= 0x40 )
    {
      v6 = 0;
      v7 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)a3 + 64 - (unsigned __int8)v4);
LABEL_5:
      v54 = (_QWORD *)(v6 | v7);
      goto LABEL_6;
    }
    goto LABEL_120;
  }
  sub_C43690((__int64)&v54, 0, 0);
  if ( !v5 )
  {
LABEL_121:
    v4 = *(_DWORD *)(a2 + 8);
    v57 = v4;
    v8 = v4;
    if ( v4 <= 0x40 )
      goto LABEL_7;
    goto LABEL_122;
  }
  if ( v5 > 0x40 )
  {
LABEL_120:
    sub_C43C90(&v54, 0, v5);
    goto LABEL_121;
  }
  v6 = (unsigned __int64)v54;
  v7 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)a3 + 64 - (unsigned __int8)v4);
  if ( v55 <= 0x40 )
  {
    v4 = *(_DWORD *)(a2 + 8);
    goto LABEL_5;
  }
  *v54 |= v7;
  v4 = *(_DWORD *)(a2 + 8);
LABEL_6:
  v57 = v4;
  v8 = v4;
  if ( v4 <= 0x40 )
  {
LABEL_7:
    v9 = v4 - 1;
    v56 = 0;
    v10 = 1LL << v9;
LABEL_8:
    v56 |= v10;
    goto LABEL_9;
  }
LABEL_122:
  v33 = v4 - 1;
  v10 = 1LL << v33;
  sub_C43690((__int64)&v56, 0, 0);
  if ( v57 <= 0x40 )
  {
    v8 = *(_DWORD *)(a2 + 8);
    v9 = v8 - 1;
    goto LABEL_8;
  }
  *(_QWORD *)(v56 + 8LL * (v33 >> 6)) |= v10;
  v8 = *(_DWORD *)(a2 + 8);
  v9 = v8 - 1;
LABEL_9:
  v59 = v8;
  v11 = ~(1LL << v9);
  if ( v8 > 0x40 )
  {
    sub_C43690((__int64)&v58, -1, 1);
    if ( v59 > 0x40 )
    {
      *(_QWORD *)(v58 + 8LL * (v9 >> 6)) &= v11;
      v71 = v55;
      if ( v55 <= 0x40 )
        goto LABEL_14;
LABEL_160:
      sub_C43780((__int64)&v70, (const void **)&v54);
      goto LABEL_15;
    }
  }
  else
  {
    v12 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v8;
    if ( !v8 )
      v12 = 0;
    v58 = v12;
  }
  v58 &= v11;
  v71 = v55;
  if ( v55 > 0x40 )
    goto LABEL_160;
LABEL_14:
  v70 = (unsigned __int64)v54;
LABEL_15:
  sub_C46A40((__int64)&v70, 1);
  v13 = v71;
  v71 = 0;
  v73 = v13;
  v72 = v70;
  sub_C46B40((__int64)&v72, (__int64 *)a2);
  v14 = v73;
  v73 = 0;
  v75 = v14;
  v74 = v72;
  sub_C4B490((__int64)&v76, (__int64)&v74, a2);
  if ( v77 > 0x40 )
  {
    sub_C43D10((__int64)&v76);
  }
  else
  {
    v15 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v77) & ~v76;
    if ( !v77 )
      v15 = 0;
    v76 = v15;
  }
  sub_C46250((__int64)&v76);
  sub_C45EE0((__int64)&v76, (__int64 *)&v54);
  v61 = v77;
  v60 = v76;
  if ( v75 > 0x40 && v74 )
    j_j___libc_free_0_0(v74);
  if ( v73 > 0x40 && v72 )
    j_j___libc_free_0_0(v72);
  if ( v71 > 0x40 && v70 )
    j_j___libc_free_0_0(v70);
  v51 = *(_DWORD *)(a2 + 8) - 1;
  v63 = 1;
  v62 = 0;
  v65 = 1;
  v64 = 0;
  v67 = 1;
  v66 = 0;
  v69 = 1;
  v68 = 0;
  sub_C4BFE0((__int64)&v56, (__int64)&v60, &v62, &v64);
  sub_C4BFE0((__int64)&v58, a2, &v66, &v68);
  while ( 1 )
  {
    ++v51;
    v75 = v61;
    if ( v61 > 0x40 )
      sub_C43780((__int64)&v74, (const void **)&v60);
    else
      v74 = v60;
    sub_C46B40((__int64)&v74, (__int64 *)&v64);
    v16 = v75;
    v75 = 0;
    v77 = v16;
    v76 = v74;
    v48 = v74;
    v17 = sub_C49970((__int64)&v64, &v76);
    if ( v16 > 0x40 )
    {
      if ( v48 )
      {
        j_j___libc_free_0_0(v48);
        if ( v75 > 0x40 )
        {
          if ( v74 )
            j_j___libc_free_0_0(v74);
        }
      }
    }
    if ( v17 >= 0 )
    {
      if ( v63 > 0x40 )
      {
        sub_C47690((__int64 *)&v62, 1u);
      }
      else
      {
        v18 = 0;
        if ( v63 >= 2 )
          v18 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v63) & (2 * v62);
        v62 = v18;
      }
      sub_C46250((__int64)&v62);
      if ( v65 > 0x40 )
      {
        sub_C47690((__int64 *)&v64, 1u);
      }
      else
      {
        v19 = 0;
        if ( v65 >= 2 )
          v19 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v65) & (2 * v64);
        v64 = v19;
      }
      sub_C46B40((__int64)&v64, (__int64 *)&v60);
LABEL_46:
      v71 = v69;
      if ( v69 <= 0x40 )
        goto LABEL_47;
      goto LABEL_142;
    }
    if ( v63 > 0x40 )
    {
      sub_C47690((__int64 *)&v62, 1u);
      v37 = v65;
      if ( v65 > 0x40 )
        goto LABEL_155;
    }
    else
    {
      v36 = 0;
      if ( v63 >= 2 )
        v36 = (2 * v62) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v63);
      v37 = v65;
      v62 = v36;
      if ( v65 > 0x40 )
      {
LABEL_155:
        sub_C47690((__int64 *)&v64, 1u);
        goto LABEL_46;
      }
    }
    v38 = 0;
    if ( v37 >= 2 )
      v38 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v37) & (2 * v64);
    v64 = v38;
    v71 = v69;
    if ( v69 <= 0x40 )
    {
LABEL_47:
      v70 = v68;
      goto LABEL_48;
    }
LABEL_142:
    sub_C43780((__int64)&v70, (const void **)&v68);
LABEL_48:
    sub_C46A40((__int64)&v70, 1);
    v20 = v71;
    v21 = v70;
    v71 = 0;
    v22 = *(_DWORD *)(a2 + 8);
    v73 = v20;
    v72 = v70;
    v75 = v22;
    if ( v22 > 0x40 )
      sub_C43780((__int64)&v74, (const void **)a2);
    else
      v74 = *(_QWORD *)a2;
    sub_C46B40((__int64)&v74, (__int64 *)&v68);
    v23 = v75;
    v75 = 0;
    v77 = v23;
    v50 = v23;
    v76 = v74;
    v47 = v74;
    v49 = sub_C49970((__int64)&v72, &v76);
    if ( v50 > 0x40 )
    {
      if ( v47 )
      {
        j_j___libc_free_0_0(v47);
        if ( v75 > 0x40 )
        {
          if ( v74 )
            j_j___libc_free_0_0(v74);
        }
      }
    }
    if ( v20 > 0x40 && v21 )
      j_j___libc_free_0_0(v21);
    if ( v71 > 0x40 && v70 )
      j_j___libc_free_0_0(v70);
    if ( v49 < 0 )
    {
      if ( (int)sub_C49970((__int64)&v66, &v56) >= 0 )
        *(_BYTE *)(a1 + 16) = 1;
      if ( v67 > 0x40 )
      {
        sub_C47690((__int64 *)&v66, 1u);
      }
      else
      {
        v34 = 0;
        if ( v67 >= 2 )
          v34 = (2 * v66) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v67);
        v66 = v34;
      }
      if ( v69 > 0x40 )
      {
        sub_C47690((__int64 *)&v68, 1u);
      }
      else
      {
        v35 = 0;
        if ( v69 >= 2 )
          v35 = (2 * v68) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v69);
        v68 = v35;
      }
      sub_C46250((__int64)&v68);
    }
    else
    {
      if ( (int)sub_C49970((__int64)&v66, &v58) >= 0 )
        *(_BYTE *)(a1 + 16) = 1;
      if ( v67 > 0x40 )
      {
        sub_C47690((__int64 *)&v66, 1u);
      }
      else
      {
        v24 = 0;
        if ( v67 >= 2 )
          v24 = (2 * v66) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v67);
        v66 = v24;
      }
      sub_C46250((__int64)&v66);
      if ( v69 > 0x40 )
      {
        sub_C47690((__int64 *)&v68, 1u);
      }
      else
      {
        v25 = 0;
        if ( v69 >= 2 )
          v25 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v69) & (2 * v68);
        v68 = v25;
      }
      sub_C46250((__int64)&v68);
      sub_C46B40((__int64)&v68, (__int64 *)a2);
    }
    if ( v53 <= 0x40 && *(_DWORD *)(a2 + 8) <= 0x40u )
    {
      v39 = *(const void **)a2;
      v53 = *(_DWORD *)(a2 + 8);
      v52 = v39;
    }
    else
    {
      sub_C43990((__int64)&v52, a2);
    }
    sub_C46E90((__int64)&v52);
    sub_C46B40((__int64)&v52, (__int64 *)&v68);
    v26 = *(_DWORD *)(a2 + 8);
    if ( 2 * v26 <= v51 )
      break;
    if ( (int)sub_C49970((__int64)&v62, (unsigned __int64 *)&v52) >= 0 )
    {
      if ( v63 <= 0x40 )
      {
        if ( (const void *)v62 != v52 )
          break;
        v27 = v65;
        if ( v65 <= 0x40 )
          goto LABEL_152;
LABEL_81:
        if ( v27 != (unsigned int)sub_C444A0((__int64)&v64) )
          break;
      }
      else
      {
        if ( !sub_C43C50((__int64)&v62, &v52) )
          break;
        v27 = v65;
        if ( v65 > 0x40 )
          goto LABEL_81;
LABEL_152:
        if ( v64 )
          break;
      }
    }
  }
  if ( !*(_BYTE *)(a1 + 16) )
    goto LABEL_86;
  _RAX = *(_QWORD *)a2;
  if ( v26 <= 0x40 )
  {
    if ( (_RAX & 1) != 0 || !a4 )
      goto LABEL_86;
    v41 = 64;
    v75 = v26;
    __asm { tzcnt   rdx, rax }
    v74 = _RAX;
    if ( _RAX )
      v41 = _RDX;
    if ( v26 <= v41 )
      v41 = v26;
LABEL_181:
    if ( v41 == v26 )
      v74 = 0;
    else
      v74 >>= v41;
    goto LABEL_166;
  }
  if ( (*(_BYTE *)_RAX & 1) != 0 || !a4 )
  {
LABEL_86:
    if ( *(_DWORD *)(a1 + 8) > 0x40u && *(_QWORD *)a1 )
      j_j___libc_free_0_0(*(_QWORD *)a1);
    *(_QWORD *)a1 = v66;
    v29 = v67;
    v67 = 0;
    *(_DWORD *)(a1 + 8) = v29;
    sub_C46250(a1);
    v30 = v51 - *(_DWORD *)(a2 + 8);
    v31 = *(_BYTE *)(a1 + 16) == 0;
    *(_DWORD *)(a1 + 20) = v30;
    if ( !v31 )
      *(_DWORD *)(a1 + 20) = v30 - 1;
    *(_DWORD *)(a1 + 24) = 0;
  }
  else
  {
    v40 = sub_C44590(a2);
    v75 = v26;
    v41 = v40;
    sub_C43780((__int64)&v74, (const void **)a2);
    v26 = v75;
    if ( v75 <= 0x40 )
      goto LABEL_181;
    sub_C482E0((__int64)&v74, v41);
LABEL_166:
    sub_3719C00(&v76, &v74, v41 + a3, 1);
    if ( *(_DWORD *)(a1 + 8) > 0x40u && *(_QWORD *)a1 )
      j_j___libc_free_0_0(*(_QWORD *)a1);
    v42 = v75 <= 0x40;
    *(_QWORD *)a1 = v76;
    v43 = v77;
    *(_DWORD *)(a1 + 24) = v41;
    *(_DWORD *)(a1 + 8) = v43;
    *(_BYTE *)(a1 + 16) = v78;
    *(_DWORD *)(a1 + 20) = v79;
    if ( !v42 && v74 )
      j_j___libc_free_0_0(v74);
  }
  if ( v69 > 0x40 && v68 )
    j_j___libc_free_0_0(v68);
  if ( v67 > 0x40 && v66 )
    j_j___libc_free_0_0(v66);
  if ( v65 > 0x40 && v64 )
    j_j___libc_free_0_0(v64);
  if ( v63 > 0x40 && v62 )
    j_j___libc_free_0_0(v62);
  if ( v61 > 0x40 && v60 )
    j_j___libc_free_0_0(v60);
  if ( v59 > 0x40 && v58 )
    j_j___libc_free_0_0(v58);
  if ( v57 > 0x40 && v56 )
    j_j___libc_free_0_0(v56);
  if ( v55 > 0x40 && v54 )
    j_j___libc_free_0_0((unsigned __int64)v54);
  if ( v53 > 0x40 && v52 )
    j_j___libc_free_0_0((unsigned __int64)v52);
  return a1;
}
