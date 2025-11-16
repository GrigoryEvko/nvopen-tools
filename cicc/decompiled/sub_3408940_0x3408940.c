// Function: sub_3408940
// Address: 0x3408940
//
__int64 __fastcall sub_3408940(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __m128i a7)
{
  __int64 v10; // rax
  __int64 v11; // r9
  __int64 v12; // rdx
  int v13; // eax
  unsigned __int16 v14; // bx
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  char v18; // al
  __int64 v19; // rsi
  _DWORD *v20; // r12
  __int64 v21; // r12
  bool v22; // al
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned int v25; // eax
  __int16 v26; // ax
  __int64 v27; // rdx
  unsigned int v28; // edx
  unsigned __int8 *v29; // r10
  unsigned __int8 *v30; // rcx
  unsigned int v31; // ebx
  __int64 v32; // rbx
  unsigned __int16 v33; // dx
  unsigned __int16 *v34; // rax
  __int64 v35; // rsi
  bool v37; // al
  __int64 v38; // rcx
  __int64 v39; // r8
  __int16 v40; // ax
  __int64 v41; // r8
  unsigned __int16 v42; // ax
  __int64 v43; // r9
  bool v44; // al
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  unsigned __int16 v48; // ax
  bool v49; // al
  __int64 v50; // r8
  bool v51; // al
  unsigned int v52; // edx
  bool v53; // al
  __int64 v54; // r8
  unsigned int v55; // edx
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rdx
  unsigned __int64 v59; // r8
  unsigned __int8 v60; // al
  __int64 v61; // rdi
  bool (__fastcall *v62)(__int64, __int64); // r8
  __int64 v63; // rax
  __int64 v64; // rdx
  unsigned int v65; // eax
  __int64 v66; // rsi
  unsigned __int8 *v67; // r10
  __int128 v68; // rax
  __int64 v69; // r9
  unsigned __int8 *v70; // rax
  unsigned __int16 v71; // ax
  __int64 v72; // rdx
  __int128 v73; // [rsp-30h] [rbp-100h]
  __int64 v74; // [rsp+10h] [rbp-C0h]
  __int64 v75; // [rsp+18h] [rbp-B8h]
  unsigned __int8 *v76; // [rsp+18h] [rbp-B8h]
  __int64 v77; // [rsp+18h] [rbp-B8h]
  unsigned int v78; // [rsp+20h] [rbp-B0h]
  int v79; // [rsp+20h] [rbp-B0h]
  unsigned __int8 *v80; // [rsp+20h] [rbp-B0h]
  unsigned int v81; // [rsp+20h] [rbp-B0h]
  unsigned int v82; // [rsp+20h] [rbp-B0h]
  unsigned __int8 *v83; // [rsp+20h] [rbp-B0h]
  unsigned __int8 *v84; // [rsp+20h] [rbp-B0h]
  unsigned __int8 *v85; // [rsp+20h] [rbp-B0h]
  int v86; // [rsp+28h] [rbp-A8h]
  int v87; // [rsp+28h] [rbp-A8h]
  unsigned int v88; // [rsp+28h] [rbp-A8h]
  unsigned int v89; // [rsp+28h] [rbp-A8h]
  __int64 v90; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v91; // [rsp+38h] [rbp-98h]
  unsigned int v92; // [rsp+40h] [rbp-90h] BYREF
  __int64 v93; // [rsp+48h] [rbp-88h]
  unsigned __int64 v94; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v95; // [rsp+58h] [rbp-78h]
  __int64 v96; // [rsp+60h] [rbp-70h]
  __int64 v97; // [rsp+68h] [rbp-68h]
  unsigned __int64 v98; // [rsp+70h] [rbp-60h] BYREF
  __int64 v99; // [rsp+78h] [rbp-58h]
  __int64 v100; // [rsp+80h] [rbp-50h] BYREF
  __int64 v101; // [rsp+88h] [rbp-48h]

  v90 = a3;
  v91 = a4;
  if ( (_WORD)a3 )
  {
    if ( (unsigned __int16)(a3 - 17) <= 0xD3u )
    {
      LOWORD(a3) = word_4456580[(unsigned __int16)a3 - 1];
      v101 = 0;
      LOWORD(v100) = a3;
      if ( !(_WORD)a3 )
        goto LABEL_5;
      goto LABEL_31;
    }
    goto LABEL_3;
  }
  v78 = a3;
  v37 = sub_30070B0((__int64)&v90);
  LOWORD(a3) = v78;
  if ( !v37 )
  {
LABEL_3:
    v10 = v91;
    goto LABEL_4;
  }
  v40 = sub_3009970((__int64)&v90, a2, v78, v38, v39);
  v41 = a3;
  LOWORD(a3) = v40;
  v10 = v41;
LABEL_4:
  LOWORD(v100) = a3;
  v101 = v10;
  if ( !(_WORD)a3 )
  {
LABEL_5:
    v96 = sub_3007260((__int64)&v100);
    LODWORD(v11) = v96;
    v97 = v12;
    goto LABEL_6;
  }
LABEL_31:
  if ( (_WORD)a3 == 1 || (unsigned __int16)(a3 - 504) <= 7u )
    goto LABEL_104;
  v11 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)a3 - 16];
LABEL_6:
  v13 = *(_DWORD *)(a1 + 24);
  if ( v13 != 35 && v13 != 11 )
  {
    v14 = v90;
    if ( (_WORD)v90 )
    {
      if ( (unsigned __int16)(v90 - 17) <= 0xD3u )
      {
        v15 = 0;
        v14 = word_4456580[(unsigned __int16)v90 - 1];
        goto LABEL_11;
      }
    }
    else
    {
      v79 = v11;
      v44 = sub_30070B0((__int64)&v90);
      LODWORD(v11) = v79;
      if ( v44 )
      {
        v48 = sub_3009970((__int64)&v90, a2, v45, v46, v47);
        LODWORD(v11) = v79;
        v14 = v48;
        goto LABEL_11;
      }
    }
    v15 = v91;
LABEL_11:
    LOWORD(v92) = v14;
    v93 = v15;
    if ( !v14 )
    {
      v86 = v11;
      v22 = sub_3007070((__int64)&v92);
      LODWORD(v11) = v86;
      if ( v22 )
        goto LABEL_46;
      v23 = sub_3007260((__int64)&v92);
      LODWORD(v11) = v86;
      v100 = v23;
      v101 = v24;
      v17 = v23;
      v18 = v101;
      goto LABEL_36;
    }
    if ( (unsigned __int16)(v14 - 2) <= 7u
      || (unsigned __int16)(v14 - 17) <= 0x6Cu
      || (unsigned __int16)(v14 - 176) <= 0x1Fu )
    {
      goto LABEL_46;
    }
    if ( v14 != 1 && (unsigned __int16)(v14 - 504) > 7u )
    {
      v16 = 16LL * (v14 - 1);
      v17 = *(_QWORD *)&byte_444C4A0[v16];
      v18 = byte_444C4A0[v16 + 8];
LABEL_36:
      v87 = v11;
      v98 = v17;
      LOBYTE(v99) = v18;
      v25 = sub_CA1930(&v98);
      LODWORD(v11) = v87;
      switch ( v25 )
      {
        case 1u:
          v26 = 2;
          v27 = 0;
          break;
        case 2u:
          v26 = 3;
          v27 = 0;
          break;
        case 4u:
          v26 = 4;
          v27 = 0;
          break;
        case 8u:
          v26 = 5;
          v27 = 0;
          break;
        case 0x10u:
          v26 = 6;
          v27 = 0;
          break;
        case 0x20u:
          v26 = 7;
          v27 = 0;
          break;
        case 0x40u:
          v26 = 8;
          v27 = 0;
          break;
        case 0x80u:
          v26 = 9;
          v27 = 0;
          break;
        default:
          v26 = sub_3007020(*(_QWORD **)(a5 + 64), v25);
          LODWORD(v11) = v87;
          break;
      }
      LOWORD(v92) = v26;
      v93 = v27;
LABEL_46:
      v88 = v11;
      v29 = sub_33FAF80(a5, 214, a6, v92, v93, v11, a7);
      v30 = v29;
      v31 = v28;
      if ( v88 > 8 )
      {
        v83 = v29;
        LODWORD(v99) = 8;
        v98 = 1;
        sub_C47700((__int64)&v94, v88, (__int64)&v98);
        v67 = v83;
        if ( (unsigned int)v99 > 0x40 && v98 )
        {
          j_j___libc_free_0_0(v98);
          v67 = v83;
        }
        v84 = v67;
        *(_QWORD *)&v68 = sub_34007B0(a5, (__int64)&v94, a6, v92, v93, 0, a7, 0);
        *((_QWORD *)&v73 + 1) = v31 | a2 & 0xFFFFFFFF00000000LL;
        *(_QWORD *)&v73 = v84;
        v70 = sub_3406EB0((_QWORD *)a5, 0x3Au, a6, v92, v93, v69, v73, v68);
        v30 = v70;
        if ( v95 > 0x40 && v94 )
        {
          v89 = v28;
          v85 = v70;
          j_j___libc_free_0_0(v94);
          v30 = v85;
          v28 = v89;
        }
      }
      v32 = v28;
      v33 = v90;
      v34 = (unsigned __int16 *)(*((_QWORD *)v30 + 6) + 16 * v32);
      v35 = *v34;
      if ( (_WORD)v90 == (_WORD)v35 )
      {
        if ( (_WORD)v90 || v91 == *((_QWORD *)v34 + 1) )
          return (__int64)v30;
        v74 = *((_QWORD *)v34 + 1);
        v76 = v30;
        v81 = (unsigned __int16)v90;
        v51 = sub_3007070((__int64)&v90);
        v50 = v74;
        v52 = v81;
        v30 = v76;
        if ( !v51 )
        {
LABEL_68:
          v77 = (__int64)v30;
          v82 = v52;
          v53 = sub_30070B0((__int64)&v90);
          v33 = v82;
          v30 = (unsigned __int8 *)v77;
          if ( v53 )
          {
            v71 = sub_3009970((__int64)&v90, v35, v82, v77, v54);
            v30 = (unsigned __int8 *)v77;
            v43 = v72;
            v33 = v71;
LABEL_70:
            v30 = sub_33FB890(a5, v33, v43, (__int64)v30, v32, a7);
            LODWORD(v32) = v55;
            v56 = *((_QWORD *)v30 + 6) + 16LL * v55;
            if ( (_WORD)v90 == *(_WORD *)v56 )
            {
              if ( (_WORD)v90 )
                return (__int64)v30;
              v50 = *(_QWORD *)(v56 + 8);
              goto LABEL_73;
            }
            return sub_32886A0(a5, (unsigned int)v90, v91, a6, (__int64)v30, v32);
          }
LABEL_69:
          v43 = v91;
          goto LABEL_70;
        }
      }
      else
      {
        if ( (_WORD)v90 )
        {
          v42 = v90 - 17;
          if ( (unsigned __int16)(v90 - 2) <= 7u || v42 <= 0x6Cu || (unsigned __int16)(v90 - 176) <= 0x1Fu )
            return sub_32886A0(a5, (unsigned int)v90, v91, a6, (__int64)v30, v32);
          if ( v42 <= 0xD3u )
          {
            v43 = 0;
            v33 = word_4456580[(unsigned __int16)v90 - 1];
            goto LABEL_70;
          }
          goto LABEL_69;
        }
        v75 = *((_QWORD *)v34 + 1);
        v80 = v30;
        v49 = sub_3007070((__int64)&v90);
        v35 = (unsigned int)v35;
        v30 = v80;
        v50 = v75;
        if ( !v49 )
        {
          v52 = 0;
          goto LABEL_68;
        }
        if ( (_WORD)v35 )
          return sub_32886A0(a5, (unsigned int)v90, v91, a6, (__int64)v30, v32);
      }
LABEL_73:
      if ( v91 == v50 )
        return (__int64)v30;
      return sub_32886A0(a5, (unsigned int)v90, v91, a6, (__int64)v30, v32);
    }
LABEL_104:
    BUG();
  }
  v19 = (unsigned int)v11;
  sub_C47700((__int64)&v94, v11, *(_QWORD *)(a1 + 96) + 24LL);
  if ( !(_WORD)v90 )
  {
    if ( !sub_3007070((__int64)&v90) )
      goto LABEL_24;
    v57 = sub_3007260((__int64)&v90);
    v99 = v58;
    v98 = v57;
LABEL_78:
    v100 = v57;
    LOBYTE(v101) = v58;
    v59 = sub_CA1930(&v100);
    v60 = 1;
    if ( v59 <= 0x40 )
    {
      v61 = *(_QWORD *)(a5 + 16);
      v62 = *(bool (__fastcall **)(__int64, __int64))(*(_QWORD *)v61 + 1336LL);
      v63 = *(_QWORD *)(a1 + 96);
      v64 = *(_QWORD *)(v63 + 24);
      v65 = *(_DWORD *)(v63 + 32);
      if ( v65 > 0x40 )
      {
        v66 = *(_QWORD *)v64;
      }
      else
      {
        v66 = 0;
        if ( v65 )
        {
          v64 <<= 64 - (unsigned __int8)v65;
          v66 = v64 >> (64 - (unsigned __int8)v65);
        }
      }
      if ( v62 == sub_2FE33F0 )
        v60 = v66 != 0;
      else
        v60 = ((__int64 (__fastcall *)(__int64, __int64, __int64))v62)(v61, v66, v64) ^ 1;
    }
    v21 = (__int64)sub_34007B0(a5, (__int64)&v94, a6, v90, v91, 0, a7, v60);
    goto LABEL_27;
  }
  if ( (unsigned __int16)(v90 - 2) <= 7u
    || (unsigned __int16)(v90 - 17) <= 0x6Cu
    || (unsigned __int16)(v90 - 176) <= 0x1Fu )
  {
    if ( (_WORD)v90 == 1 || (unsigned __int16)(v90 - 504) <= 7u )
      goto LABEL_104;
    v58 = 16LL * ((unsigned __int16)v90 - 1);
    v57 = *(_QWORD *)&byte_444C4A0[v58];
    LOBYTE(v58) = byte_444C4A0[v58 + 8];
    goto LABEL_78;
  }
LABEL_24:
  v20 = sub_300AC80((unsigned __int16 *)&v90, v19);
  if ( v20 == sub_C33340() )
    sub_C3C640(&v100, (__int64)v20, &v94);
  else
    sub_C3B160((__int64)&v100, v20, (__int64 *)&v94);
  v21 = sub_33FE6E0(a5, &v100, a6, v90, v91, 0, a7);
  sub_91D830(&v100);
LABEL_27:
  if ( v95 > 0x40 && v94 )
    j_j___libc_free_0_0(v94);
  return v21;
}
