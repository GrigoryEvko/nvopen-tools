// Function: sub_DD0560
// Address: 0xdd0560
//
__int64 __fastcall sub_DD0560(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int *v5; // rax
  unsigned int v6; // r8d
  unsigned int v7; // eax
  unsigned __int64 v8; // rdx
  const void *v9; // rdx
  unsigned int v10; // eax
  __int64 v11; // rsi
  const void *v12; // rax
  __int64 v13; // rsi
  char v14; // al
  unsigned int v16; // eax
  unsigned __int64 v17; // rdx
  const void *v18; // rdx
  unsigned int v19; // eax
  __int64 v20; // rsi
  unsigned int v21; // r8d
  __int64 v22; // rsi
  char v23; // al
  unsigned int v24; // eax
  __int64 *v25; // rax
  __int64 *v26; // r12
  __int64 v27; // rbx
  unsigned int v28; // edx
  unsigned int v29; // eax
  __int64 *v30; // rax
  __int64 v31; // rax
  _QWORD *v32; // rax
  _QWORD *v33; // rax
  unsigned int v34; // eax
  __int64 *v35; // rax
  __int64 v36; // rax
  __int64 v37; // rsi
  _QWORD *v38; // rax
  _QWORD *v39; // rax
  char v40; // al
  const void **v41; // rsi
  __int64 *v42; // rax
  __int64 v43; // rax
  _QWORD *v44; // rax
  _QWORD *v45; // rax
  unsigned int v46; // eax
  __int64 *v47; // rax
  __int64 v48; // r13
  _QWORD *v49; // rax
  _QWORD *v50; // rax
  bool v51; // zf
  char v52; // al
  char v53; // al
  unsigned int v54; // eax
  const void *v55; // rax
  __int64 v56; // [rsp+8h] [rbp-108h]
  __int64 v57; // [rsp+8h] [rbp-108h]
  __int64 v58; // [rsp+8h] [rbp-108h]
  unsigned int v59; // [rsp+10h] [rbp-100h]
  __int64 *v60; // [rsp+10h] [rbp-100h]
  unsigned int v61; // [rsp+18h] [rbp-F8h]
  __int64 v62; // [rsp+18h] [rbp-F8h]
  __int64 v63; // [rsp+18h] [rbp-F8h]
  unsigned int v64; // [rsp+20h] [rbp-F0h]
  unsigned int v65; // [rsp+20h] [rbp-F0h]
  const void *v66; // [rsp+30h] [rbp-E0h] BYREF
  unsigned int v67; // [rsp+38h] [rbp-D8h]
  __int128 v68; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v69; // [rsp+50h] [rbp-C0h]
  const void *v70; // [rsp+60h] [rbp-B0h] BYREF
  unsigned int v71; // [rsp+68h] [rbp-A8h]
  char v72; // [rsp+70h] [rbp-A0h]
  const void *v73; // [rsp+80h] [rbp-90h] BYREF
  unsigned int v74; // [rsp+88h] [rbp-88h]
  char v75; // [rsp+90h] [rbp-80h]
  const void *v76; // [rsp+A0h] [rbp-70h] BYREF
  unsigned int v77; // [rsp+A8h] [rbp-68h]
  char v78; // [rsp+B0h] [rbp-60h]
  const void *v79; // [rsp+C0h] [rbp-50h] BYREF
  unsigned int v80; // [rsp+C8h] [rbp-48h]
  char v81; // [rsp+D0h] [rbp-40h]

  sub_C47360(a3, *(__int64 **)a2);
  v5 = *(unsigned int **)(a2 + 8);
  v69 = 0;
  v68 = 0;
  v6 = *v5;
  if ( *v5 <= 1 )
    goto LABEL_2;
  v16 = *(_DWORD *)(a3 + 8);
  v67 = v16;
  if ( v16 > 0x40 )
  {
    v65 = v6;
    sub_C43780((__int64)&v66, (const void **)a3);
    v16 = v67;
    v6 = v65;
    if ( v67 > 0x40 )
    {
      sub_C43D10((__int64)&v66);
      v6 = v65;
      goto LABEL_35;
    }
    v17 = (unsigned __int64)v66;
  }
  else
  {
    v17 = *(_QWORD *)a3;
  }
  v18 = (const void *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v16) & ~v17);
  if ( !v16 )
    v18 = 0;
  v66 = v18;
LABEL_35:
  v64 = v6;
  sub_C46250((__int64)&v66);
  v19 = v67;
  v20 = *(_QWORD *)(a2 + 24);
  v67 = 0;
  v21 = v64;
  v71 = v19;
  v70 = v66;
  v77 = *(_DWORD *)(v20 + 8);
  if ( v77 > 0x40 )
  {
    sub_C43780((__int64)&v76, (const void **)v20);
    v21 = v64;
  }
  else
  {
    v76 = *(const void **)v20;
  }
  v22 = *(_QWORD *)(a2 + 16);
  v74 = *(_DWORD *)(v22 + 8);
  if ( v74 > 0x40 )
  {
    v61 = v21;
    sub_C43780((__int64)&v73, (const void **)v22);
    v21 = v61;
  }
  else
  {
    v73 = *(const void **)v22;
  }
  sub_C4CD10((__int64)&v79, &v73, (__int64 *)&v76, (unsigned __int64 *)&v70, v21);
  if ( (_BYTE)v69 )
  {
    if ( v81 )
    {
      if ( DWORD2(v68) <= 0x40 || !(_QWORD)v68 )
      {
        *(_QWORD *)&v68 = v79;
        v29 = v80;
        v80 = 0;
        DWORD2(v68) = v29;
        goto LABEL_66;
      }
      j_j___libc_free_0_0(v68);
      v23 = v81;
      *(_QWORD *)&v68 = v79;
      v28 = v80;
      v80 = 0;
      DWORD2(v68) = v28;
    }
    else
    {
      LOBYTE(v69) = 0;
      sub_969240((__int64 *)&v68);
      v23 = v81;
    }
    if ( !v23 )
      goto LABEL_43;
LABEL_66:
    v81 = 0;
    if ( v80 > 0x40 && v79 )
      j_j___libc_free_0_0(v79);
    goto LABEL_43;
  }
  if ( v81 )
  {
    v24 = v80;
    LOBYTE(v69) = 1;
    v80 = 0;
    DWORD2(v68) = v24;
    *(_QWORD *)&v68 = v79;
    goto LABEL_66;
  }
LABEL_43:
  if ( v74 > 0x40 && v73 )
    j_j___libc_free_0_0(v73);
  if ( v77 > 0x40 && v76 )
    j_j___libc_free_0_0(v76);
  if ( v71 > 0x40 && v70 )
    j_j___libc_free_0_0(v70);
  if ( v67 > 0x40 && v66 )
    j_j___libc_free_0_0(v66);
  v6 = **(_DWORD **)(a2 + 8);
LABEL_2:
  v59 = v6 + 1;
  v7 = *(_DWORD *)(a3 + 8);
  v67 = v7;
  if ( v7 > 0x40 )
  {
    sub_C43780((__int64)&v66, (const void **)a3);
    v7 = v67;
    if ( v67 > 0x40 )
    {
      sub_C43D10((__int64)&v66);
      goto LABEL_7;
    }
    v8 = (unsigned __int64)v66;
  }
  else
  {
    v8 = *(_QWORD *)a3;
  }
  v9 = (const void *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v7) & ~v8);
  if ( !v7 )
    v9 = 0;
  v66 = v9;
LABEL_7:
  sub_C46250((__int64)&v66);
  v10 = v67;
  v11 = *(_QWORD *)(a2 + 24);
  v67 = 0;
  v74 = v10;
  v73 = v66;
  v80 = *(_DWORD *)(v11 + 8);
  if ( v80 > 0x40 )
  {
    sub_C43780((__int64)&v79, (const void **)v11);
    v13 = *(_QWORD *)(a2 + 16);
    v77 = *(_DWORD *)(v13 + 8);
    if ( v77 <= 0x40 )
      goto LABEL_9;
LABEL_57:
    sub_C43780((__int64)&v76, (const void **)v13);
    goto LABEL_10;
  }
  v12 = *(const void **)v11;
  v13 = *(_QWORD *)(a2 + 16);
  v79 = v12;
  v77 = *(_DWORD *)(v13 + 8);
  if ( v77 > 0x40 )
    goto LABEL_57;
LABEL_9:
  v76 = *(const void **)v13;
LABEL_10:
  sub_C4CD10((__int64)&v70, &v76, (__int64 *)&v79, (unsigned __int64 *)&v73, v59);
  if ( v77 > 0x40 && v76 )
    j_j___libc_free_0_0(v76);
  if ( v80 > 0x40 && v79 )
    j_j___libc_free_0_0(v79);
  if ( v74 > 0x40 && v73 )
    j_j___libc_free_0_0(v73);
  if ( v67 > 0x40 && v66 )
    j_j___libc_free_0_0(v66);
  v14 = v72;
  if ( (_BYTE)v69 && v72 )
  {
    v25 = *(__int64 **)(a2 + 40);
    v26 = *(__int64 **)(a2 + 32);
    v81 = 0;
    v27 = *(_QWORD *)(a2 + 48);
    v60 = v25;
    v80 = v71;
    if ( v71 > 0x40 )
    {
      sub_C43780((__int64)&v79, &v70);
      v81 = 1;
      v78 = 0;
      if ( !(_BYTE)v69 )
        goto LABEL_83;
    }
    else
    {
      v81 = 1;
      v78 = 0;
      v79 = v70;
    }
    v77 = DWORD2(v68);
    if ( DWORD2(v68) > 0x40 )
      sub_C43780((__int64)&v76, (const void **)&v68);
    else
      v76 = (const void *)v68;
    v78 = 1;
LABEL_83:
    sub_D92C00((__int64)&v73, (__int64)&v76, (__int64)&v79);
    if ( v78 )
    {
      v78 = 0;
      sub_969240((__int64 *)&v76);
    }
    if ( v81 )
    {
      v81 = 0;
      sub_969240((__int64 *)&v79);
    }
    v30 = (__int64 *)sub_B2BE50(*v26);
    v31 = sub_ACCFD0(v30, (__int64)&v73);
    v56 = *v60;
    v32 = sub_DA2570((__int64)v26, v31);
    v33 = sub_DD0540(v56, (__int64)v32, v26);
    if ( !sub_AB1B10(v27, v33[4] + 24LL) )
    {
      v80 = v74;
      if ( v74 > 0x40 )
        sub_C43780((__int64)&v79, &v73);
      else
        v79 = v73;
      sub_C46F20((__int64)&v79, 1u);
      v34 = v80;
      v80 = 0;
      v77 = v34;
      v76 = v79;
      v35 = (__int64 *)sub_B2BE50(*v26);
      v36 = sub_ACCFD0(v35, (__int64)&v76);
      v37 = v36;
      if ( v77 > 0x40 && v76 )
      {
        v57 = v36;
        j_j___libc_free_0_0(v76);
        v37 = v57;
      }
      if ( v80 > 0x40 && v79 )
        j_j___libc_free_0_0(v79);
      v58 = *v60;
      v38 = sub_DA2570((__int64)v26, v37);
      v39 = sub_DD0540(v58, (__int64)v38, v26);
      if ( sub_AB1B10(v27, v39[4] + 24LL) )
      {
        v51 = v75 == 0;
        *(_BYTE *)(a1 + 16) = 0;
        if ( v51 )
        {
          *(_BYTE *)(a1 + 24) = 1;
LABEL_106:
          v14 = v72;
          goto LABEL_25;
        }
        sub_9865C0(a1, (__int64)&v73);
        *(_BYTE *)(a1 + 16) = 1;
        v52 = v75;
        *(_BYTE *)(a1 + 24) = 1;
LABEL_104:
        if ( v52 )
        {
          v75 = 0;
          sub_969240((__int64 *)&v73);
        }
        goto LABEL_106;
      }
    }
    v40 = v69;
    v41 = (const void **)&v68;
    if ( v75 != (_BYTE)v69 )
    {
LABEL_96:
      v81 = 0;
      if ( !v40 )
      {
LABEL_97:
        v42 = (__int64 *)sub_B2BE50(*v26);
        v43 = sub_ACCFD0(v42, (__int64)&v79);
        v62 = *v60;
        v44 = sub_DA2570((__int64)v26, v43);
        v45 = sub_DD0540(v62, (__int64)v44, v26);
        if ( sub_AB1B10(v27, v45[4] + 24LL) )
          goto LABEL_107;
        v77 = v80;
        if ( v80 > 0x40 )
          sub_C43780((__int64)&v76, &v79);
        else
          v76 = v79;
        sub_C46F20((__int64)&v76, 1u);
        v46 = v77;
        v77 = 0;
        v67 = v46;
        v66 = v76;
        v47 = (__int64 *)sub_B2BE50(*v26);
        v63 = sub_ACCFD0(v47, (__int64)&v66);
        sub_969240((__int64 *)&v66);
        sub_969240((__int64 *)&v76);
        v48 = *v60;
        v49 = sub_DA2570((__int64)v26, v63);
        v50 = sub_DD0540(v48, (__int64)v49, v26);
        if ( !sub_AB1B10(v27, v50[4] + 24LL) )
        {
LABEL_107:
          *(_BYTE *)(a1 + 16) = 0;
          *(_BYTE *)(a1 + 24) = 1;
          v53 = v81;
        }
        else
        {
          v51 = v81 == 0;
          *(_BYTE *)(a1 + 16) = 0;
          if ( v51 )
          {
            *(_BYTE *)(a1 + 24) = 1;
LABEL_103:
            v52 = v75;
            goto LABEL_104;
          }
          v54 = v80;
          *(_DWORD *)(a1 + 8) = v80;
          if ( v54 <= 0x40 )
          {
            v55 = v79;
            *(_BYTE *)(a1 + 16) = 1;
            *(_BYTE *)(a1 + 24) = 1;
            *(_QWORD *)a1 = v55;
            goto LABEL_109;
          }
          sub_C43780(a1, &v79);
          *(_BYTE *)(a1 + 16) = 1;
          v53 = v81;
          *(_BYTE *)(a1 + 24) = 1;
        }
        if ( !v53 )
          goto LABEL_103;
LABEL_109:
        v81 = 0;
        sub_969240((__int64 *)&v79);
        goto LABEL_103;
      }
LABEL_121:
      sub_9865C0((__int64)&v79, (__int64)v41);
      v81 = 1;
      goto LABEL_97;
    }
    if ( v75 )
    {
      if ( v74 <= 0x40 )
      {
        if ( v73 != (const void *)v68 )
          goto LABEL_120;
      }
      else if ( !sub_C43C50((__int64)&v73, (const void **)&v68) )
      {
LABEL_120:
        v81 = 0;
        v41 = (const void **)&v68;
        goto LABEL_121;
      }
    }
    v40 = v72;
    v41 = &v70;
    goto LABEL_96;
  }
  *(_BYTE *)(a1 + 16) = 0;
  *(_BYTE *)(a1 + 24) = 0;
LABEL_25:
  if ( v14 )
  {
    v72 = 0;
    if ( v71 > 0x40 )
    {
      if ( v70 )
        j_j___libc_free_0_0(v70);
    }
  }
  if ( (_BYTE)v69 )
  {
    LOBYTE(v69) = 0;
    if ( DWORD2(v68) > 0x40 )
    {
      if ( (_QWORD)v68 )
        j_j___libc_free_0_0(v68);
    }
  }
  return a1;
}
