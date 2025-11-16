// Function: sub_326F3F0
// Address: 0x326f3f0
//
__int64 __fastcall sub_326F3F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8,
        __int64 a9)
{
  unsigned __int16 v11; // bx
  __int64 result; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rbx
  __int64 v18; // rdx
  unsigned __int16 v19; // cx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned int v23; // edx
  bool v24; // al
  __int64 v25; // rdx
  __int64 v26; // r8
  unsigned __int16 v27; // bx
  __int64 v28; // rdx
  __int64 v29; // r8
  __int64 v30; // rdx
  unsigned __int16 v31; // bx
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  unsigned __int64 v35; // rax
  __int128 v36; // rax
  int v37; // r9d
  unsigned int v38; // edx
  int v39; // r9d
  unsigned int v40; // edx
  int v41; // r9d
  __int64 v42; // rax
  unsigned int v43; // edx
  int v44; // r9d
  bool v45; // al
  __int64 v46; // rdx
  __int64 v47; // rcx
  unsigned __int16 v48; // ax
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int128 v52; // [rsp-20h] [rbp-140h]
  __int128 v53; // [rsp-20h] [rbp-140h]
  __int128 v54; // [rsp-20h] [rbp-140h]
  __int128 v55; // [rsp-10h] [rbp-130h]
  __int128 v56; // [rsp-10h] [rbp-130h]
  unsigned int v57; // [rsp+8h] [rbp-118h]
  __int64 v58; // [rsp+10h] [rbp-110h]
  int v59; // [rsp+10h] [rbp-110h]
  __int128 v60; // [rsp+10h] [rbp-110h]
  int v61; // [rsp+10h] [rbp-110h]
  __int64 v62; // [rsp+30h] [rbp-F0h]
  __int64 v63; // [rsp+40h] [rbp-E0h]
  __int64 v64; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v65; // [rsp+58h] [rbp-C8h]
  __int64 v66; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v67; // [rsp+68h] [rbp-B8h]
  unsigned __int64 v68; // [rsp+70h] [rbp-B0h] BYREF
  unsigned int v69; // [rsp+78h] [rbp-A8h]
  unsigned __int16 v70; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v71; // [rsp+88h] [rbp-98h]
  unsigned __int16 v72; // [rsp+90h] [rbp-90h] BYREF
  __int64 v73; // [rsp+98h] [rbp-88h]
  __int64 v74; // [rsp+A0h] [rbp-80h]
  __int64 v75; // [rsp+A8h] [rbp-78h]
  __int64 v76; // [rsp+B0h] [rbp-70h]
  __int64 v77; // [rsp+B8h] [rbp-68h]
  unsigned __int64 v78; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v79; // [rsp+C8h] [rbp-58h]
  __int64 v80; // [rsp+D0h] [rbp-50h]
  __int64 v81; // [rsp+D8h] [rbp-48h]
  __int64 v82; // [rsp+E0h] [rbp-40h] BYREF
  __int64 v83; // [rsp+E8h] [rbp-38h]
  unsigned __int64 v84; // [rsp+138h] [rbp+18h]
  unsigned __int64 v85; // [rsp+138h] [rbp+18h]

  v11 = a1;
  v66 = a1;
  v67 = a2;
  v64 = a3;
  v65 = a4;
  if ( (_WORD)a1 == (_WORD)a3 )
  {
    if ( (_WORD)a1 || a2 == a4 )
    {
      *((_QWORD *)&v52 + 1) = a6;
      *(_QWORD *)&v52 = a5;
      return sub_3406EB0(a8, 85, a9, v66, v67, a6, v52, a7);
    }
  }
  else if ( (_WORD)a1 )
  {
    if ( (unsigned __int16)(a1 - 17) <= 0xD3u )
    {
      v13 = 0;
      v11 = word_4456580[(unsigned __int16)a1 - 1];
      goto LABEL_11;
    }
    goto LABEL_10;
  }
  if ( !sub_30070B0((__int64)&v66) )
  {
LABEL_10:
    v13 = v67;
    goto LABEL_11;
  }
  v11 = sub_3009970((__int64)&v66, a2, v14, v15, v16);
LABEL_11:
  v72 = v11;
  v73 = v13;
  if ( v11 )
  {
    if ( v11 == 1 || (unsigned __int16)(v11 - 504) <= 7u )
      goto LABEL_76;
    v17 = *(_QWORD *)&byte_444C4A0[16 * v11 - 16];
  }
  else
  {
    v74 = sub_3007260((__int64)&v72);
    LODWORD(v17) = v74;
    v75 = v18;
  }
  v19 = v64;
  if ( (_WORD)v64 )
  {
    if ( (unsigned __int16)(v64 - 17) > 0xD3u )
    {
LABEL_15:
      v20 = v65;
      goto LABEL_16;
    }
    v19 = word_4456580[(unsigned __int16)v64 - 1];
    v20 = 0;
  }
  else
  {
    v24 = sub_30070B0((__int64)&v64);
    v19 = 0;
    if ( !v24 )
      goto LABEL_15;
    v19 = sub_3009970((__int64)&v64, a2, v25, 0, v26);
  }
LABEL_16:
  v70 = v19;
  v71 = v20;
  if ( v19 )
  {
    if ( v19 == 1 || (unsigned __int16)(v19 - 504) <= 7u )
      goto LABEL_76;
    v21 = *(_QWORD *)&byte_444C4A0[16 * v19 - 16];
  }
  else
  {
    v21 = sub_3007260((__int64)&v70);
    v76 = v21;
    v77 = v22;
  }
  v69 = v21;
  v23 = v21;
  if ( (unsigned int)v21 > 0x40 )
  {
    sub_C43690((__int64)&v68, 0, 0);
    v23 = v69;
  }
  else
  {
    v68 = 0;
  }
  if ( (_DWORD)v17 != v23 )
  {
    if ( (unsigned int)v17 > 0x3F || v23 > 0x40 )
      sub_C43C90(&v68, v17, v23);
    else
      v68 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v17 + 64 - (unsigned __int8)v23) << v17;
  }
  if ( (unsigned __int8)sub_33DD210(a8, a5, a6, &v68, 0) )
  {
    v27 = v66;
    if ( (_WORD)v66 )
    {
      if ( (unsigned __int16)(v66 - 17) > 0xD3u )
      {
LABEL_37:
        v28 = v67;
        goto LABEL_38;
      }
      v27 = word_4456580[(unsigned __int16)v66 - 1];
      v28 = 0;
    }
    else
    {
      if ( !sub_30070B0((__int64)&v66) )
        goto LABEL_37;
      v27 = sub_3009970((__int64)&v66, a5, v49, v50, v51);
    }
LABEL_38:
    LOWORD(v82) = v27;
    v83 = v28;
    if ( v27 )
    {
      if ( v27 == 1 || (unsigned __int16)(v27 - 504) <= 7u )
        goto LABEL_76;
      v29 = *(_QWORD *)&byte_444C4A0[16 * v27 - 16];
    }
    else
    {
      v80 = sub_3007260((__int64)&v82);
      LODWORD(v29) = v80;
      v81 = v30;
    }
    v31 = v64;
    if ( (_WORD)v64 )
    {
      if ( (unsigned __int16)(v64 - 17) <= 0xD3u )
      {
        v31 = word_4456580[(unsigned __int16)v64 - 1];
        v32 = 0;
        goto LABEL_43;
      }
    }
    else
    {
      v57 = v29;
      v45 = sub_30070B0((__int64)&v64);
      LODWORD(v29) = v57;
      if ( v45 )
      {
        v48 = sub_3009970((__int64)&v64, a5, v46, v47, v57);
        LODWORD(v29) = v57;
        v31 = v48;
        goto LABEL_43;
      }
    }
    v32 = v65;
LABEL_43:
    LOWORD(v78) = v31;
    v79 = v32;
    if ( !v31 )
    {
      v59 = v29;
      v33 = sub_3007260((__int64)&v78);
      LODWORD(v29) = v59;
      v82 = v33;
      v83 = v34;
LABEL_45:
      LODWORD(v79) = v33;
      if ( (unsigned int)v33 > 0x40 )
      {
        v61 = v29;
        sub_C43690((__int64)&v78, 0, 0);
        LODWORD(v29) = v61;
      }
      else
      {
        v78 = 0;
      }
      if ( (_DWORD)v29 )
      {
        if ( (unsigned int)v29 > 0x40 )
        {
          sub_C43C90(&v78, 0, v29);
        }
        else
        {
          v35 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v29);
          if ( (unsigned int)v79 > 0x40 )
            *(_QWORD *)v78 |= v35;
          else
            v78 |= v35;
        }
      }
      *(_QWORD *)&v36 = sub_34007B0(a8, (unsigned int)&v78, a9, v64, v65, 0, 0);
      if ( (unsigned int)v79 > 0x40 && v78 )
      {
        v60 = v36;
        j_j___libc_free_0_0(v78);
        v36 = v60;
      }
      v63 = sub_3406EB0(a8, 182, a9, v64, v65, v37, a7, v36);
      v84 = v38 | *((_QWORD *)&a7 + 1) & 0xFFFFFFFF00000000LL;
      *((_QWORD *)&v55 + 1) = v84;
      *(_QWORD *)&v55 = v63;
      *((_QWORD *)&v53 + 1) = a6;
      v62 = sub_33FAF80(a8, 216, a9, v66, v67, v39, v55);
      *(_QWORD *)&v53 = a5;
      v85 = v40 | v84 & 0xFFFFFFFF00000000LL;
      v42 = sub_33FAF80(a8, 216, a9, v66, v67, v41, v53);
      *((_QWORD *)&v56 + 1) = v85;
      *(_QWORD *)&v56 = v62;
      *((_QWORD *)&v54 + 1) = v43 | a6 & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v54 = v42;
      result = sub_3406EB0(a8, 85, a9, v66, v67, v44, v54, v56);
      goto LABEL_26;
    }
    if ( v31 != 1 && (unsigned __int16)(v31 - 504) > 7u )
    {
      v33 = *(_QWORD *)&byte_444C4A0[16 * v31 - 16];
      goto LABEL_45;
    }
LABEL_76:
    BUG();
  }
  result = 0;
LABEL_26:
  if ( v69 > 0x40 )
  {
    if ( v68 )
    {
      v58 = result;
      j_j___libc_free_0_0(v68);
      return v58;
    }
  }
  return result;
}
