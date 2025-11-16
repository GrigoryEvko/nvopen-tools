// Function: sub_20CCB60
// Address: 0x20ccb60
//
__int64 __fastcall sub_20CCB60(
        unsigned int a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        double a7,
        double a8,
        double a9)
{
  __int64 v13; // r15
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r13
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 **v22; // r11
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r9
  __int64 v27; // rax
  __int64 v28; // r9
  __int64 **v29; // rdx
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 v32; // r9
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r9
  __int64 v36; // rax
  __int64 v37; // rdi
  __int64 *v38; // rbx
  __int64 v39; // rax
  __int64 v40; // rcx
  __int64 v41; // rsi
  __int64 v42; // rsi
  unsigned __int8 *v43; // rsi
  __int64 v44; // rax
  __int64 v45; // rdi
  __int64 v46; // r9
  __int64 *v47; // r13
  __int64 v48; // rcx
  __int64 v49; // rax
  __int64 v50; // rsi
  __int64 v51; // rsi
  __int64 v52; // r13
  unsigned __int8 *v53; // rsi
  __int64 v54; // rax
  __int64 v55; // rdi
  __int64 v56; // r9
  __int64 v57; // rax
  __int64 v58; // rsi
  __int64 v59; // rsi
  __int64 v60; // rsi
  __int64 v61; // rdx
  unsigned __int8 *v62; // rsi
  __int64 v63; // rax
  __int64 v64; // rdi
  __int64 v65; // r9
  __int64 v66; // rsi
  __int64 v67; // rax
  __int64 v68; // rsi
  __int64 v69; // rsi
  __int64 v70; // rdx
  unsigned __int8 *v71; // rsi
  __int64 v72; // rax
  __int64 v73; // rdi
  __int64 *v74; // rbx
  __int64 v75; // rax
  __int64 v76; // rcx
  __int64 *v77; // [rsp+10h] [rbp-C0h]
  __int64 v78; // [rsp+18h] [rbp-B8h]
  __int64 v79; // [rsp+18h] [rbp-B8h]
  __int64 v80; // [rsp+18h] [rbp-B8h]
  __int64 v81; // [rsp+18h] [rbp-B8h]
  __int64 v83; // [rsp+20h] [rbp-B0h]
  __int64 v84; // [rsp+20h] [rbp-B0h]
  __int64 v85; // [rsp+20h] [rbp-B0h]
  __int64 v86; // [rsp+20h] [rbp-B0h]
  __int64 v87; // [rsp+20h] [rbp-B0h]
  __int64 *v88; // [rsp+20h] [rbp-B0h]
  __int64 v89; // [rsp+20h] [rbp-B0h]
  __int64 v90; // [rsp+20h] [rbp-B0h]
  __int64 **v91; // [rsp+28h] [rbp-A8h]
  __int64 v92; // [rsp+38h] [rbp-98h] BYREF
  __int64 v93[2]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v94; // [rsp+50h] [rbp-80h]
  __int64 v95[2]; // [rsp+60h] [rbp-70h] BYREF
  __int16 v96; // [rsp+70h] [rbp-60h]
  _BYTE v97[16]; // [rsp+80h] [rbp-50h] BYREF
  __int16 v98; // [rsp+90h] [rbp-40h]

  if ( a1 > 6 )
  {
    v22 = *(__int64 ***)(a6 + 8);
    v94 = 257;
    v24 = *(_QWORD *)(a6 + 24);
    v91 = v22;
    v96 = 257;
    v25 = sub_156E320(a2, a3, v24, (__int64)v93, 0);
    v26 = v25;
    if ( v91 != *(__int64 ***)v25 )
    {
      if ( *(_BYTE *)(v25 + 16) > 0x10u )
      {
        v98 = 257;
        v54 = sub_15FDBD0(36, v25, (__int64)v91, (__int64)v97, 0);
        v55 = a2[1];
        v56 = v54;
        if ( v55 )
        {
          v78 = v54;
          v77 = (__int64 *)a2[2];
          sub_157E9D0(v55 + 40, v54);
          v56 = v78;
          v57 = *(_QWORD *)(v78 + 24);
          v58 = *v77;
          *(_QWORD *)(v78 + 32) = v77;
          v58 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v78 + 24) = v58 | v57 & 7;
          *(_QWORD *)(v58 + 8) = v78 + 24;
          *v77 = *v77 & 7 | (v78 + 24);
        }
        v79 = v56;
        sub_164B780(v56, v95);
        v59 = *a2;
        v26 = v79;
        if ( *a2 )
        {
          v92 = *a2;
          sub_1623A60((__int64)&v92, v59, 2);
          v26 = v79;
          v60 = *(_QWORD *)(v79 + 48);
          v61 = v79 + 48;
          if ( v60 )
          {
            sub_161E7C0(v79 + 48, v60);
            v26 = v79;
            v61 = v79 + 48;
          }
          v62 = (unsigned __int8 *)v92;
          *(_QWORD *)(v26 + 48) = v92;
          if ( v62 )
          {
            v80 = v26;
            sub_1623210((__int64)&v92, v62, v61);
            v26 = v80;
          }
        }
      }
      else
      {
        v26 = sub_15A46C0(36, (__int64 ***)v25, v91, 0);
      }
    }
    v27 = sub_20CC690(a1, a2, v26, a5, a7, a8, a9);
    v28 = *(_QWORD *)(a6 + 24);
    v29 = *(__int64 ***)a6;
    v96 = 257;
    v30 = v27;
    v94 = 257;
    if ( v29 != *(__int64 ***)v27 )
    {
      v83 = v28;
      if ( *(_BYTE *)(v27 + 16) > 0x10u )
      {
        v98 = 257;
        v63 = sub_15FDBD0(37, v27, (__int64)v29, (__int64)v97, 0);
        v64 = a2[1];
        v65 = v83;
        v30 = v63;
        if ( v64 )
        {
          v81 = v83;
          v88 = (__int64 *)a2[2];
          sub_157E9D0(v64 + 40, v63);
          v65 = v81;
          v66 = *v88;
          v67 = *(_QWORD *)(v30 + 24) & 7LL;
          *(_QWORD *)(v30 + 32) = v88;
          v66 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v30 + 24) = v66 | v67;
          *(_QWORD *)(v66 + 8) = v30 + 24;
          *v88 = *v88 & 7 | (v30 + 24);
        }
        v89 = v65;
        sub_164B780(v30, v93);
        v68 = *a2;
        v28 = v89;
        if ( *a2 )
        {
          v92 = *a2;
          sub_1623A60((__int64)&v92, v68, 2);
          v69 = *(_QWORD *)(v30 + 48);
          v70 = v30 + 48;
          v28 = v89;
          if ( v69 )
          {
            sub_161E7C0(v30 + 48, v69);
            v28 = v89;
            v70 = v30 + 48;
          }
          v71 = (unsigned __int8 *)v92;
          *(_QWORD *)(v30 + 48) = v92;
          if ( v71 )
          {
            v90 = v28;
            sub_1623210((__int64)&v92, v71, v70);
            v28 = v90;
          }
        }
      }
      else
      {
        v31 = sub_15A46C0(37, (__int64 ***)v27, v29, 0);
        v28 = v83;
        v30 = v31;
      }
    }
    if ( *(_BYTE *)(v30 + 16) > 0x10u || *(_BYTE *)(v28 + 16) > 0x10u )
    {
      v98 = 257;
      v44 = sub_15FB440(23, (__int64 *)v30, v28, (__int64)v97, 0);
      v45 = a2[1];
      v46 = v44;
      if ( v45 )
      {
        v47 = (__int64 *)a2[2];
        v85 = v44;
        sub_157E9D0(v45 + 40, v44);
        v46 = v85;
        v48 = *v47;
        v49 = *(_QWORD *)(v85 + 24);
        *(_QWORD *)(v85 + 32) = v47;
        v48 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v85 + 24) = v48 | v49 & 7;
        *(_QWORD *)(v48 + 8) = v85 + 24;
        *v47 = *v47 & 7 | (v85 + 24);
      }
      v86 = v46;
      sub_164B780(v46, v95);
      v50 = *a2;
      v32 = v86;
      if ( *a2 )
      {
        v92 = *a2;
        sub_1623A60((__int64)&v92, v50, 2);
        v32 = v86;
        v51 = *(_QWORD *)(v86 + 48);
        v52 = v86 + 48;
        if ( v51 )
        {
          sub_161E7C0(v86 + 48, v51);
          v32 = v86;
        }
        v53 = (unsigned __int8 *)v92;
        *(_QWORD *)(v32 + 48) = v92;
        if ( v53 )
        {
          v87 = v32;
          sub_1623210((__int64)&v92, v53, v52);
          v32 = v87;
        }
      }
    }
    else
    {
      v32 = sub_15A2D50((__int64 *)v30, v28, 0, 0, a7, a8, a9);
    }
    v98 = 257;
    v84 = v32;
    v33 = sub_1281C00(a2, a3, *(_QWORD *)(a6 + 40), (__int64)v97);
    v35 = v84;
    v96 = 257;
    v17 = v33;
    if ( *(_BYTE *)(v84 + 16) <= 0x10u )
    {
      if ( sub_1593BB0(v84, a3, v34, 257) )
        return v17;
      v35 = v84;
      if ( *(_BYTE *)(v17 + 16) <= 0x10u )
        return sub_15A2D10((__int64 *)v17, v84, a7, a8, a9);
    }
    v98 = 257;
    v72 = sub_15FB440(27, (__int64 *)v17, v35, (__int64)v97, 0);
    v73 = a2[1];
    v17 = v72;
    if ( v73 )
    {
      v74 = (__int64 *)a2[2];
      sub_157E9D0(v73 + 40, v72);
      v75 = *(_QWORD *)(v17 + 24);
      v76 = *v74;
      *(_QWORD *)(v17 + 32) = v74;
      v76 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v17 + 24) = v76 | v75 & 7;
      *(_QWORD *)(v76 + 8) = v17 + 24;
      *v74 = *v74 & 7 | (v17 + 24);
    }
    sub_164B780(v17, v95);
    v41 = *a2;
    if ( !*a2 )
      return v17;
    v93[0] = *a2;
    goto LABEL_28;
  }
  v13 = a4;
  if ( a1 <= 4 )
  {
    if ( a1 )
    {
      v19 = sub_20CC690(a1, a2, a3, a4, a7, a8, a9);
      v20 = *(_QWORD *)(a6 + 32);
      v98 = 257;
      v21 = sub_1281C00(a2, v19, v20, (__int64)v97);
      v14 = *(_QWORD *)(a6 + 40);
      v98 = 257;
      v13 = v21;
    }
    else
    {
      v14 = *(_QWORD *)(a6 + 40);
      v98 = 257;
    }
    v17 = sub_1281C00(a2, a3, v14, (__int64)v97);
    v96 = 257;
    if ( *(_BYTE *)(v13 + 16) <= 0x10u )
    {
      if ( sub_1593BB0(v13, a3, v15, v16) )
        return v17;
      if ( *(_BYTE *)(v17 + 16) <= 0x10u )
        return sub_15A2D10((__int64 *)v17, v13, a7, a8, a9);
    }
    v98 = 257;
    v36 = sub_15FB440(27, (__int64 *)v17, v13, (__int64)v97, 0);
    v37 = a2[1];
    v17 = v36;
    if ( v37 )
    {
      v38 = (__int64 *)a2[2];
      sub_157E9D0(v37 + 40, v36);
      v39 = *(_QWORD *)(v17 + 24);
      v40 = *v38;
      *(_QWORD *)(v17 + 32) = v38;
      v40 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v17 + 24) = v40 | v39 & 7;
      *(_QWORD *)(v40 + 8) = v17 + 24;
      *v38 = *v38 & 7 | (v17 + 24);
    }
    sub_164B780(v17, v95);
    v41 = *a2;
    if ( !*a2 )
      return v17;
    v93[0] = *a2;
LABEL_28:
    sub_1623A60((__int64)v93, v41, 2);
    v42 = *(_QWORD *)(v17 + 48);
    if ( v42 )
      sub_161E7C0(v17 + 48, v42);
    v43 = (unsigned __int8 *)v93[0];
    *(_QWORD *)(v17 + 48) = v93[0];
    if ( v43 )
      sub_1623210((__int64)v93, v43, v17 + 48);
    return v17;
  }
  return sub_20CC690(a1, a2, a3, a4, a7, a8, a9);
}
