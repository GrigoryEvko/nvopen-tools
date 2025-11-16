// Function: sub_3277840
// Address: 0x3277840
//
__int64 __fastcall sub_3277840(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5, __int64 a6, char a7)
{
  unsigned __int64 v7; // rax
  __int64 v9; // r10
  __int64 v10; // r9
  __int64 v11; // r14
  unsigned int v13; // r13d
  unsigned __int64 v14; // r12
  unsigned __int16 *v17; // rdx
  int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rax
  _QWORD *v21; // rax
  __int64 v22; // rsi
  __int64 v23; // r15
  __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // r10
  int v27; // r11d
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 *v31; // r12
  int v32; // ebx
  _QWORD *v33; // r12
  unsigned int v34; // ebx
  int v35; // eax
  unsigned __int64 v36; // rdi
  unsigned __int16 *v37; // rdx
  int v38; // eax
  __int64 v39; // rdx
  bool v40; // al
  __int64 v41; // rcx
  __int64 v42; // rax
  __int64 v43; // rdx
  unsigned __int64 v44; // r14
  __int64 v45; // rax
  int v46; // edx
  int v47; // esi
  __int64 v48; // rax
  __int64 v49; // rax
  bool v50; // al
  __int64 v51; // rcx
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rdx
  _QWORD *v56; // rax
  int v57; // ebx
  unsigned __int64 v58; // rax
  _QWORD *v59; // rax
  __int64 v61; // [rsp+10h] [rbp-C0h]
  __int64 v62; // [rsp+10h] [rbp-C0h]
  char v63; // [rsp+18h] [rbp-B8h]
  __int64 v64; // [rsp+18h] [rbp-B8h]
  __int64 v65; // [rsp+18h] [rbp-B8h]
  int v66; // [rsp+20h] [rbp-B0h]
  __int64 v67; // [rsp+20h] [rbp-B0h]
  __int64 v68; // [rsp+20h] [rbp-B0h]
  __int64 v69; // [rsp+20h] [rbp-B0h]
  int v70; // [rsp+20h] [rbp-B0h]
  __int64 v71; // [rsp+20h] [rbp-B0h]
  __int64 v72; // [rsp+28h] [rbp-A8h]
  __int64 v73; // [rsp+28h] [rbp-A8h]
  int v74; // [rsp+28h] [rbp-A8h]
  int v75; // [rsp+28h] [rbp-A8h]
  __int64 v76; // [rsp+28h] [rbp-A8h]
  __int64 v77; // [rsp+28h] [rbp-A8h]
  __int64 v78; // [rsp+28h] [rbp-A8h]
  __int64 v79; // [rsp+28h] [rbp-A8h]
  __int64 v80; // [rsp+28h] [rbp-A8h]
  __int64 v81; // [rsp+30h] [rbp-A0h]
  __int64 v82; // [rsp+30h] [rbp-A0h]
  __int64 v83; // [rsp+30h] [rbp-A0h]
  __int64 v84; // [rsp+30h] [rbp-A0h]
  __int64 v85; // [rsp+30h] [rbp-A0h]
  __int64 v86; // [rsp+30h] [rbp-A0h]
  unsigned int v87; // [rsp+3Ch] [rbp-94h]
  __int64 v88; // [rsp+40h] [rbp-90h]
  int v89; // [rsp+48h] [rbp-88h]
  __int16 v90; // [rsp+50h] [rbp-80h] BYREF
  __int64 v91; // [rsp+58h] [rbp-78h]
  __int64 v92; // [rsp+60h] [rbp-70h]
  __int64 v93; // [rsp+68h] [rbp-68h]
  _QWORD *v94; // [rsp+70h] [rbp-60h] BYREF
  __int64 v95; // [rsp+78h] [rbp-58h]
  unsigned __int64 v96; // [rsp+80h] [rbp-50h] BYREF
  __int64 v97; // [rsp+88h] [rbp-48h]
  unsigned __int64 v98; // [rsp+90h] [rbp-40h] BYREF
  __int64 v99; // [rsp+98h] [rbp-38h]

  v7 = a5;
  v9 = a4;
  v10 = a3;
  v11 = a1;
  v87 = v7;
  v89 = a2;
  v88 = *(_QWORD *)(a6 + 16);
  if ( !a7 || !v7 || (v7 & (v7 - 1)) != 0 )
    goto LABEL_3;
  _BitScanReverse64(&v7, v7);
  v17 = (unsigned __int16 *)(*(_QWORD *)(a3 + 48) + 16LL * (unsigned int)a4);
  v63 = v7 ^ 0x3F;
  v13 = 63 - (v7 ^ 0x3F);
  v18 = *v17;
  v19 = *((_QWORD *)v17 + 1);
  v90 = v18;
  v91 = v19;
  if ( (_WORD)v18 )
  {
    if ( (unsigned __int16)(v18 - 17) > 0xD3u )
    {
      LOWORD(v98) = v18;
      v99 = v19;
      goto LABEL_11;
    }
    LOWORD(v18) = word_4456580[v18 - 1];
    v54 = 0;
  }
  else
  {
    v62 = a6;
    v68 = v10;
    v76 = v19;
    v50 = sub_30070B0((__int64)&v90);
    v10 = v68;
    a6 = v62;
    v9 = a4;
    if ( !v50 )
    {
      v99 = v76;
      LOWORD(v98) = 0;
LABEL_56:
      v69 = v9;
      v77 = a6;
      v84 = v10;
      v20 = sub_3007260((__int64)&v98);
      v9 = v69;
      a6 = v77;
      v92 = v20;
      v10 = v84;
      v93 = v52;
      goto LABEL_57;
    }
    LOWORD(v18) = sub_3009970((__int64)&v90, a2, v76, v51, v62);
    v9 = a4;
    a6 = v62;
    v10 = v68;
  }
  LOWORD(v98) = v18;
  v99 = v54;
  if ( !(_WORD)v18 )
    goto LABEL_56;
LABEL_11:
  if ( (_WORD)v18 == 1 || (unsigned __int16)(v18 - 504) <= 7u )
    goto LABEL_100;
  v20 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v18 - 16];
LABEL_57:
  if ( (unsigned int)v20 < v13 )
  {
LABEL_3:
    v13 = 0;
    goto LABEL_4;
  }
  LODWORD(v99) = v20;
  if ( (unsigned int)v20 > 0x40 )
  {
    v71 = v9;
    v80 = a6;
    sub_C43690((__int64)&v98, 0, 0);
    a6 = v80;
    v9 = v71;
    if ( !v13 )
      goto LABEL_60;
    v58 = 0xFFFFFFFFFFFFFFFFLL >> (v63 + 1);
    if ( (unsigned int)v99 > 0x40 )
    {
      *(_QWORD *)v98 |= v58;
      goto LABEL_60;
    }
LABEL_93:
    v98 |= v58;
    goto LABEL_60;
  }
  v98 = 0;
  if ( v13 )
  {
    v58 = 0xFFFFFFFFFFFFFFFFLL >> (v63 + 1);
    goto LABEL_93;
  }
LABEL_60:
  v85 = a6;
  v53 = sub_34494D0(v88, a3, v9, &v98, a6, 0);
  a6 = v85;
  v10 = v53;
  if ( !v53 )
  {
    v10 = a3;
    v13 = 0;
  }
  if ( (unsigned int)v99 > 0x40 && v98 )
  {
    v78 = v85;
    v86 = v10;
    j_j___libc_free_0_0(v98);
    a6 = v78;
    v10 = v86;
  }
LABEL_4:
  if ( *(_DWORD *)(v10 + 24) != 57
    || (v21 = *(_QWORD **)(v10 + 40), v72 = a6, v81 = v10, v22 = v21[1], (v23 = sub_33DFBC0(*v21, v22, 0, 0)) == 0) )
  {
    LODWORD(v14) = 0;
    return (unsigned int)v14;
  }
  v24 = v72;
  v25 = *(_QWORD *)(v81 + 40);
  v26 = *(_QWORD *)(v25 + 40);
  v27 = *(_DWORD *)(v25 + 48);
  if ( v13 )
  {
    v37 = (unsigned __int16 *)(*(_QWORD *)(a1 + 48) + 16LL * (unsigned int)a2);
    v38 = *v37;
    v39 = *((_QWORD *)v37 + 1);
    LOWORD(v94) = v38;
    v95 = v39;
    if ( (_WORD)v38 )
    {
      if ( (unsigned __int16)(v38 - 17) > 0xD3u )
      {
        LOWORD(v96) = v38;
        v97 = v39;
LABEL_67:
        if ( (_WORD)v38 != 1 && (unsigned __int16)(v38 - 504) > 7u )
        {
          v42 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v38 - 16];
LABEL_39:
          if ( v13 > (unsigned int)v42 )
            goto LABEL_16;
          LODWORD(v97) = v42;
          v44 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v13);
          if ( (unsigned int)v42 > 0x40 )
          {
            v65 = v24;
            v70 = v27;
            v79 = v26;
            sub_C43690((__int64)&v96, 0, 0);
            v26 = v79;
            v27 = v70;
            v24 = v65;
            if ( (unsigned int)v97 > 0x40 )
            {
              *(_QWORD *)v96 |= v44;
              goto LABEL_43;
            }
          }
          else
          {
            v96 = 0;
          }
          v96 |= v44;
LABEL_43:
          v75 = v27;
          v83 = v26;
          v45 = sub_34494D0(v88, a1, a2, &v96, v24, 0);
          v26 = v83;
          v27 = v75;
          v47 = v46;
          if ( !v45 )
          {
            v47 = v89;
            v45 = a1;
          }
          v89 = v47;
          v11 = v45;
          if ( (unsigned int)v97 > 0x40 && v96 )
          {
            j_j___libc_free_0_0(v96);
            v27 = v75;
            v26 = v83;
          }
          goto LABEL_16;
        }
LABEL_100:
        BUG();
      }
      LOWORD(v38) = word_4456580[v38 - 1];
      v55 = 0;
    }
    else
    {
      v61 = v72;
      v64 = v39;
      v66 = v27;
      v73 = v26;
      v40 = sub_30070B0((__int64)&v94);
      v26 = v73;
      v27 = v66;
      v24 = v61;
      if ( !v40 )
      {
        v97 = v64;
        LOWORD(v96) = 0;
LABEL_38:
        v67 = v24;
        v74 = v27;
        v82 = v26;
        v42 = sub_3007260((__int64)&v96);
        v24 = v67;
        v27 = v74;
        v98 = v42;
        v26 = v82;
        v99 = v43;
        goto LABEL_39;
      }
      LOWORD(v38) = sub_3009970((__int64)&v94, v22, v64, v41, v61);
      v24 = v61;
      v27 = v66;
      v26 = v73;
    }
    LOWORD(v96) = v38;
    v97 = v55;
    if ( !(_WORD)v38 )
      goto LABEL_38;
    goto LABEL_67;
  }
LABEL_16:
  LODWORD(v95) = 1;
  v94 = 0;
  if ( v26 == v11 && v27 == v89
    || *(_DWORD *)(v26 + 24) == 216
    && (v48 = *(_QWORD *)(v26 + 40), *(_QWORD *)v48 == v11)
    && *(_DWORD *)(v48 + 8) == v89 )
  {
    v49 = *(_QWORD *)(v23 + 96);
    v34 = *(_DWORD *)(v49 + 32);
    if ( v34 <= 0x40 )
    {
      v59 = *(_QWORD **)(v49 + 24);
      v14 = v87;
      LODWORD(v95) = v34;
      v94 = v59;
      if ( !v13 )
        goto LABEL_96;
LABEL_87:
      v14 = (unsigned __int64)&v96;
      sub_C443A0((__int64)&v96, (__int64)&v94, v13);
      v57 = v97;
      if ( (unsigned int)v97 <= 0x40 )
      {
        LOBYTE(v14) = v96 == 0;
        goto LABEL_80;
      }
      if ( v57 - (unsigned int)sub_C444A0((__int64)&v96) <= 0x40 )
      {
        LOBYTE(v14) = *(_QWORD *)v96 == 0;
      }
      else
      {
        LODWORD(v14) = 0;
        if ( !v96 )
        {
LABEL_80:
          v34 = v95;
LABEL_81:
          if ( v34 <= 0x40 )
            return (unsigned int)v14;
          v36 = (unsigned __int64)v94;
          goto LABEL_33;
        }
      }
      j_j___libc_free_0_0(v96);
      goto LABEL_80;
    }
    sub_C43990((__int64)&v94, v49 + 24);
  }
  else
  {
    LODWORD(v14) = 0;
    if ( *(_DWORD *)(v11 + 24) != 56 )
      return (unsigned int)v14;
    v28 = *(_QWORD *)(v11 + 40);
    if ( v26 != *(_QWORD *)v28 || v27 != *(_DWORD *)(v28 + 8) )
      return (unsigned int)v14;
    v29 = sub_33DFBC0(*(_QWORD *)(v28 + 40), *(_QWORD *)(v28 + 48), 0, 0);
    if ( !v29 )
      goto LABEL_80;
    v30 = *(_QWORD *)(v29 + 96);
    v31 = (__int64 *)(*(_QWORD *)(v23 + 96) + 24LL);
    LODWORD(v97) = *(_DWORD *)(v30 + 32);
    if ( (unsigned int)v97 > 0x40 )
      sub_C43780((__int64)&v96, (const void **)(v30 + 24));
    else
      v96 = *(_QWORD *)(v30 + 24);
    sub_C45EE0((__int64)&v96, v31);
    v32 = v97;
    LODWORD(v97) = 0;
    v33 = (_QWORD *)v96;
    if ( (unsigned int)v95 > 0x40 && v94 )
    {
      j_j___libc_free_0_0((unsigned __int64)v94);
      v94 = v33;
      LODWORD(v95) = v32;
      if ( (unsigned int)v97 > 0x40 && v96 )
        j_j___libc_free_0_0(v96);
    }
    else
    {
      v94 = (_QWORD *)v96;
      LODWORD(v95) = v32;
    }
  }
  if ( v13 )
    goto LABEL_87;
  v34 = v95;
  v14 = v87;
  if ( (unsigned int)v95 <= 0x40 )
  {
LABEL_96:
    v56 = v94;
    goto LABEL_84;
  }
  v35 = sub_C444A0((__int64)&v94);
  v36 = (unsigned __int64)v94;
  if ( v34 - v35 <= 0x40 )
  {
    v56 = (_QWORD *)*v94;
LABEL_84:
    LOBYTE(v14) = v14 == (_QWORD)v56;
    goto LABEL_81;
  }
  LODWORD(v14) = 0;
LABEL_33:
  if ( v36 )
    j_j___libc_free_0_0(v36);
  return (unsigned int)v14;
}
