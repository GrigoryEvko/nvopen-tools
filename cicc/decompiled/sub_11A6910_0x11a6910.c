// Function: sub_11A6910
// Address: 0x11a6910
//
__int64 __fastcall sub_11A6910(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6, __int64 a7)
{
  __int64 *v10; // rdx
  unsigned int v11; // eax
  unsigned __int64 **v12; // r8
  unsigned __int64 v13; // rcx
  unsigned int v14; // ebx
  unsigned __int64 v15; // rsi
  __int64 v16; // r8
  int v17; // eax
  _QWORD *v18; // rax
  unsigned int v19; // eax
  unsigned int v20; // r12d
  unsigned int v21; // edx
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rax
  unsigned __int64 v26; // rax
  __int128 v27; // rax
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rax
  unsigned int v32; // ebx
  unsigned int v33; // r15d
  unsigned __int64 v34; // rax
  unsigned int v35; // esi
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // rcx
  __int64 v38; // rax
  unsigned __int64 v39; // r15
  int v40; // eax
  unsigned int v41; // esi
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rax
  unsigned __int64 v44; // r15
  unsigned __int64 v45; // r8
  unsigned __int64 v46; // r8
  bool v47; // r14
  unsigned int v48; // eax
  unsigned __int64 v49; // rdx
  unsigned __int64 v50; // rdx
  unsigned int v51; // r15d
  unsigned int v52; // eax
  __int64 v53; // rdi
  __int64 v54; // rax
  char v55; // al
  char v56; // al
  _QWORD *v57; // r8
  signed __int64 v58; // rsi
  __int64 v59; // rsi
  __int64 v60; // r14
  unsigned __int8 *v61; // rsi
  __int64 v62; // rdi
  __int64 v63; // rax
  __int64 v64; // r8
  bool v65; // al
  unsigned __int64 v66; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v67; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v68; // [rsp+10h] [rbp-E0h]
  char v70; // [rsp+20h] [rbp-D0h]
  unsigned __int64 **v71; // [rsp+20h] [rbp-D0h]
  __int64 v73; // [rsp+30h] [rbp-C0h]
  unsigned int v75; // [rsp+38h] [rbp-B8h]
  unsigned int v76; // [rsp+38h] [rbp-B8h]
  unsigned int v77; // [rsp+38h] [rbp-B8h]
  __int64 v78; // [rsp+38h] [rbp-B8h]
  __int64 v79; // [rsp+38h] [rbp-B8h]
  unsigned __int8 *v80; // [rsp+38h] [rbp-B8h]
  _QWORD *v81; // [rsp+38h] [rbp-B8h]
  _QWORD *v82; // [rsp+38h] [rbp-B8h]
  _QWORD *v83; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v84; // [rsp+38h] [rbp-B8h]
  __int64 v85; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v86; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v87; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v88; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v89; // [rsp+58h] [rbp-98h]
  unsigned __int64 v90; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v91; // [rsp+68h] [rbp-88h]
  unsigned __int64 v92; // [rsp+70h] [rbp-80h] BYREF
  unsigned int v93; // [rsp+78h] [rbp-78h]
  unsigned __int64 v94; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v95; // [rsp+88h] [rbp-68h]
  unsigned __int64 v96; // [rsp+90h] [rbp-60h] BYREF
  unsigned int v97; // [rsp+98h] [rbp-58h]
  __int16 v98; // [rsp+B0h] [rbp-40h]

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v10 = *(__int64 **)(a2 - 8);
  else
    v10 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v73 = *v10;
  v11 = sub_BCB060(*(_QWORD *)(*v10 + 8));
  v12 = (unsigned __int64 **)a5;
  v13 = v11;
  v14 = v11;
  v75 = *(_DWORD *)(a5 + 8);
  if ( v75 > 0x40 )
  {
    v67 = v11;
    v71 = v12;
    v40 = sub_C444A0((__int64)v12);
    v13 = v67;
    if ( v75 - v40 > 0x40 )
      return 0;
    v16 = 0;
    v15 = **v71;
    if ( v67 <= v15 )
      return v16;
  }
  else
  {
    v15 = (unsigned __int64)*v12;
    v16 = 0;
    if ( v11 <= v15 )
      return v16;
  }
  v76 = *(_DWORD *)(a3 + 8);
  if ( v76 > 0x40 )
  {
    v66 = v13;
    v17 = sub_C444A0(a3);
    v13 = v66;
    if ( v76 - v17 <= 0x40 )
    {
      v18 = **(_QWORD ***)a3;
      goto LABEL_8;
    }
    return 0;
  }
  v18 = *(_QWORD **)a3;
LABEL_8:
  v16 = 0;
  if ( v13 <= (unsigned __int64)v18 )
    return v16;
  v77 = (unsigned int)v18;
  v19 = *(_DWORD *)(a7 + 24);
  v20 = v15;
  if ( v19 > 0x40 )
    memset(*(void **)(a7 + 16), 0, 8 * (((unsigned __int64)v19 + 63) >> 6));
  else
    *(_QWORD *)(a7 + 16) = 0;
  v21 = v15 - 1;
  if ( (_DWORD)v15 != 1 )
  {
    if ( v21 > 0x40 )
    {
      sub_C43C90((_QWORD *)a7, 0, v21);
    }
    else
    {
      v22 = 0xFFFFFFFFFFFFFFFFLL >> (65 - (unsigned __int8)v15);
      v23 = *(_QWORD *)a7;
      if ( *(_DWORD *)(a7 + 8) <= 0x40u )
      {
        v24 = v22 | v23;
        *(_QWORD *)a7 = v24;
LABEL_21:
        *(_QWORD *)a7 = *a6 & v24;
        goto LABEL_22;
      }
      *(_QWORD *)v23 |= v22;
    }
  }
  if ( *(_DWORD *)(a7 + 8) <= 0x40u )
  {
    v24 = *(_QWORD *)a7;
    goto LABEL_21;
  }
  sub_C43B90((_QWORD *)a7, a6);
LABEL_22:
  v87 = v14;
  if ( v14 <= 0x40 )
  {
    v26 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v14;
    if ( v14 )
    {
      v86 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v14;
      v89 = v14;
    }
    else
    {
      v86 = 0;
      v26 = 0;
      v89 = 0;
    }
    v88 = v26;
    v70 = *(_BYTE *)a2;
    if ( *(_BYTE *)a2 != 55 )
    {
      v95 = v14;
      goto LABEL_29;
    }
    v95 = v14;
LABEL_103:
    v94 = v86;
    v48 = v95;
    goto LABEL_104;
  }
  sub_C43690((__int64)&v86, -1, 1);
  v89 = v14;
  sub_C43690((__int64)&v88, -1, 1);
  v14 = v87;
  v70 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 != 55 )
  {
    v95 = v87;
    if ( v87 > 0x40 )
    {
      sub_C43780((__int64)&v94, (const void **)&v86);
      v14 = v95;
      if ( v95 > 0x40 )
      {
        sub_C44B70((__int64)&v94, v77);
        v14 = v95;
        v97 = v95;
        if ( v95 > 0x40 )
        {
          sub_C43780((__int64)&v96, (const void **)&v94);
          v14 = v97;
          if ( v97 > 0x40 )
          {
            sub_C47690((__int64 *)&v96, v15);
            goto LABEL_43;
          }
LABEL_38:
          v30 = 0;
          if ( (_DWORD)v15 != v14 )
            v30 = v96 << v15;
          v31 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v14) & v30;
          if ( !v14 )
            v31 = 0;
          v96 = v31;
LABEL_43:
          if ( v87 > 0x40 && v86 )
            j_j___libc_free_0_0(v86);
          v86 = v96;
          v87 = v97;
          if ( v95 > 0x40 && v94 )
            j_j___libc_free_0_0(v94);
          v32 = v89;
          if ( (unsigned int)v15 < v77 )
          {
            v97 = v89;
            v33 = v77 - v15;
            if ( v89 <= 0x40 )
            {
              v34 = v88;
              v35 = v89;
LABEL_50:
              v36 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v32;
              v37 = 0;
              if ( v32 )
              {
                v38 = (__int64)(v34 << (64 - (unsigned __int8)v32)) >> (64 - (unsigned __int8)v32);
                if ( v33 == v32 )
                  v37 = v36 & (v38 >> 63);
                else
                  v37 = v36 & (v38 >> v33);
              }
              v96 = v37;
              goto LABEL_54;
            }
            sub_C43780((__int64)&v96, (const void **)&v88);
            v32 = v97;
            if ( v97 <= 0x40 )
            {
              v34 = v96;
              v35 = v89;
              goto LABEL_50;
            }
            sub_C44B70((__int64)&v96, v33);
            v35 = v89;
LABEL_54:
            if ( v35 > 0x40 && v88 )
              j_j___libc_free_0_0(v88);
            v32 = v97;
            v88 = v96;
            v89 = v97;
            goto LABEL_58;
          }
          goto LABEL_66;
        }
LABEL_37:
        v96 = v94;
        goto LABEL_38;
      }
      *((_QWORD *)&v27 + 1) = v94;
LABEL_30:
      *(_QWORD *)&v27 = 0;
      if ( v14 )
        *(_QWORD *)&v27 = (__int64)(*((_QWORD *)&v27 + 1) << (64 - (unsigned __int8)v14)) >> (64 - (unsigned __int8)v14);
      v27 = (__int64)v27;
      v97 = v14;
      *(_QWORD *)&v27 = (__int64)v27 >> v77;
      if ( v77 == v14 )
        *(_QWORD *)&v27 = *((_QWORD *)&v27 + 1);
      v29 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v14) & v27;
      if ( !v14 )
        v29 = 0;
      v94 = v29;
      goto LABEL_37;
    }
LABEL_29:
    *((_QWORD *)&v27 + 1) = v86;
    goto LABEL_30;
  }
  v95 = v87;
  if ( v87 <= 0x40 )
    goto LABEL_103;
  sub_C43780((__int64)&v94, (const void **)&v86);
  v14 = v95;
  v48 = v95;
  if ( v95 > 0x40 )
  {
    sub_C482E0((__int64)&v94, v77);
    v48 = v95;
    goto LABEL_106;
  }
LABEL_104:
  if ( v77 == v14 )
    v94 = 0;
  else
    v94 >>= v77;
LABEL_106:
  v97 = v48;
  if ( v48 <= 0x40 )
  {
    v96 = v94;
LABEL_108:
    v49 = 0;
    if ( (_DWORD)v15 != v48 )
      v49 = v96 << v15;
    v50 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v48) & v49;
    if ( !v48 )
      v50 = 0;
    v96 = v50;
    goto LABEL_113;
  }
  sub_C43780((__int64)&v96, (const void **)&v94);
  v48 = v97;
  if ( v97 <= 0x40 )
    goto LABEL_108;
  sub_C47690((__int64 *)&v96, v15);
LABEL_113:
  if ( v87 > 0x40 && v86 )
    j_j___libc_free_0_0(v86);
  v86 = v96;
  v87 = v97;
  if ( v95 > 0x40 && v94 )
    j_j___libc_free_0_0(v94);
  v32 = v89;
  if ( (unsigned int)v15 < v77 )
  {
    v97 = v89;
    v51 = v77 - v15;
    if ( v89 > 0x40 )
    {
      sub_C43780((__int64)&v96, (const void **)&v88);
      v32 = v97;
      if ( v97 > 0x40 )
      {
        sub_C482E0((__int64)&v96, v51);
        v70 = 55;
        v35 = v89;
        goto LABEL_54;
      }
    }
    else
    {
      v96 = v88;
    }
    if ( v32 == v51 )
    {
      v96 = 0;
      v35 = v89;
      v70 = 55;
    }
    else
    {
      v70 = 55;
      v35 = v89;
      v96 >>= v51;
    }
    goto LABEL_54;
  }
  v70 = 55;
LABEL_66:
  v41 = v15 - v77;
  if ( v32 <= 0x40 )
  {
    v42 = 0;
    if ( v41 != v32 )
      v42 = v88 << v41;
    v43 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v32) & v42;
    if ( !v32 )
      v43 = 0;
    v88 = v43;
    goto LABEL_72;
  }
  sub_C47690((__int64 *)&v88, v41);
  v32 = v89;
LABEL_58:
  v95 = v32;
  if ( v32 <= 0x40 )
  {
LABEL_72:
    v44 = v88;
    goto LABEL_73;
  }
  sub_C43780((__int64)&v94, (const void **)&v88);
  v32 = v95;
  if ( v95 > 0x40 )
  {
    sub_C43B90(&v94, a6);
    v32 = v95;
    v39 = v94;
    goto LABEL_74;
  }
  v44 = v94;
LABEL_73:
  v39 = *a6 & v44;
  v94 = v39;
LABEL_74:
  v97 = v32;
  v96 = v39;
  v95 = 0;
  v91 = v87;
  if ( v87 <= 0x40 )
  {
    v45 = v86;
LABEL_76:
    v46 = *a6 & v45;
LABEL_77:
    v47 = v46 == v39;
    goto LABEL_78;
  }
  sub_C43780((__int64)&v90, (const void **)&v86);
  if ( v91 <= 0x40 )
  {
    v45 = v90;
    goto LABEL_76;
  }
  sub_C43B90(&v90, a6);
  v52 = v91;
  v46 = v90;
  v91 = 0;
  v93 = v52;
  v92 = v90;
  if ( v52 <= 0x40 )
    goto LABEL_77;
  v68 = v90;
  v47 = sub_C43C50((__int64)&v92, (const void **)&v96);
  if ( v68 )
  {
    j_j___libc_free_0_0(v68);
    if ( v91 > 0x40 )
    {
      if ( v90 )
        j_j___libc_free_0_0(v90);
    }
  }
LABEL_78:
  if ( v32 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  if ( v95 > 0x40 && v94 )
    j_j___libc_free_0_0(v94);
  v16 = 0;
  if ( !v47 )
    goto LABEL_87;
  v16 = v73;
  if ( v20 == v77 )
    goto LABEL_87;
  v16 = *(_QWORD *)(a2 + 16);
  if ( !v16 )
    goto LABEL_87;
  if ( *(_QWORD *)(v16 + 8) )
  {
    v16 = 0;
    goto LABEL_87;
  }
  v53 = *(_QWORD *)(v73 + 8);
  if ( v20 <= v77 )
  {
    v63 = sub_AD64C0(v53, v77 - v20, 0);
    v98 = 257;
    if ( v70 == 55 )
      v64 = sub_B504D0(26, v73, v63, (__int64)&v96, 0, 0);
    else
      v64 = sub_B504D0(27, v73, v63, (__int64)&v96, 0, 0);
    v85 = v64;
    v65 = sub_B44E60(a2);
    v57 = (_QWORD *)v85;
    if ( v65 )
    {
      sub_B448B0(v85, 1);
      v57 = (_QWORD *)v85;
    }
  }
  else
  {
    v54 = sub_AD64C0(v53, v20 - v77, 0);
    v98 = 257;
    v80 = (unsigned __int8 *)sub_B504D0(25, v73, v54, (__int64)&v96, 0, 0);
    v55 = sub_B44900(a4);
    sub_B44850(v80, v55);
    v56 = sub_B448F0(a4);
    sub_B447F0(v80, v56);
    v57 = v80;
  }
  v58 = *(_QWORD *)(a4 + 48);
  v96 = v58;
  if ( v58 )
  {
    v81 = v57;
    sub_B96E90((__int64)&v96, v58, 1);
    v57 = v81;
    v59 = v81[6];
    v60 = (__int64)(v81 + 6);
    if ( !v59 )
      goto LABEL_150;
    goto LABEL_149;
  }
  v59 = v57[6];
  v60 = (__int64)(v57 + 6);
  if ( v59 )
  {
LABEL_149:
    v82 = v57;
    sub_B91220(v60, v59);
    v57 = v82;
LABEL_150:
    v61 = (unsigned __int8 *)v96;
    v57[6] = v96;
    if ( v61 )
    {
      v83 = v57;
      sub_B976B0((__int64)&v96, v61, v60);
      v57 = v83;
    }
  }
  v84 = (unsigned __int64)v57;
  sub_B44220(v57, a4 + 24, 0);
  v62 = *(_QWORD *)(a1 + 40);
  v96 = v84;
  sub_11A2F60(v62 + 2096, (__int64 *)&v96);
  v16 = v84;
LABEL_87:
  if ( v89 > 0x40 && v88 )
  {
    v78 = v16;
    j_j___libc_free_0_0(v88);
    v16 = v78;
  }
  if ( v87 > 0x40 && v86 )
  {
    v79 = v16;
    j_j___libc_free_0_0(v86);
    return v79;
  }
  return v16;
}
