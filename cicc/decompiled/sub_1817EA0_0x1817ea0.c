// Function: sub_1817EA0
// Address: 0x1817ea0
//
unsigned __int8 *__fastcall sub_1817EA0(__int64 **a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v6; // rax
  unsigned __int8 *v7; // rsi
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rsi
  __int64 v16; // rdi
  __int64 **v17; // rax
  __int64 **v18; // r9
  __int64 v19; // rax
  int v20; // edx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // r12
  _QWORD *v25; // rbx
  unsigned int v26; // r13d
  __int64 *v27; // rax
  unsigned int v28; // ebx
  __int64 *v29; // rax
  unsigned __int8 *result; // rax
  __int64 *v31; // rax
  __int64 *v32; // rax
  __int64 v33; // rax
  __int64 v34; // rbx
  __int64 *v35; // rax
  unsigned __int8 *v36; // rax
  __int64 *v37; // rax
  __int64 v38; // rsi
  __int64 *v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rbx
  __int64 *v42; // rax
  __int64 v43; // rax
  __int64 *v44; // r15
  __int64 v45; // rcx
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rsi
  __int64 v49; // rax
  __int64 v50; // rsi
  __int64 v51; // rdx
  unsigned __int8 *v52; // rsi
  __int64 v53; // rax
  __int64 **v54; // r9
  __int64 v55; // rsi
  __int64 v56; // rax
  __int64 v57; // rsi
  __int64 v58; // rdx
  unsigned __int8 *v59; // rsi
  __int64 *v60; // rax
  __int64 v61; // rax
  __int64 v62; // r13
  __int64 *v63; // rax
  __int64 **v64; // [rsp+10h] [rbp-F0h]
  __int64 **v66; // [rsp+18h] [rbp-E8h]
  __int64 *v67; // [rsp+18h] [rbp-E8h]
  __int64 **v68; // [rsp+18h] [rbp-E8h]
  __int64 *v69; // [rsp+18h] [rbp-E8h]
  __int64 **v70; // [rsp+18h] [rbp-E8h]
  __int64 **v71; // [rsp+18h] [rbp-E8h]
  __int64 v72; // [rsp+20h] [rbp-E0h]
  unsigned __int8 *v73; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v74[2]; // [rsp+40h] [rbp-C0h] BYREF
  __int16 v75; // [rsp+50h] [rbp-B0h]
  unsigned __int8 *v76[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v77; // [rsp+70h] [rbp-90h]
  __int64 v78; // [rsp+78h] [rbp-88h]
  unsigned __int8 *v79; // [rsp+80h] [rbp-80h] BYREF
  __int64 v80; // [rsp+88h] [rbp-78h]
  __int64 *v81; // [rsp+90h] [rbp-70h]
  __int64 v82; // [rsp+98h] [rbp-68h]
  __int64 v83; // [rsp+A0h] [rbp-60h]
  int v84; // [rsp+A8h] [rbp-58h]
  __int64 v85; // [rsp+B0h] [rbp-50h]
  __int64 v86; // [rsp+B8h] [rbp-48h]

  v6 = sub_16498A0(a2);
  v7 = *(unsigned __int8 **)(a2 + 48);
  v79 = 0;
  v82 = v6;
  v8 = *(_QWORD *)(a2 + 40);
  v83 = 0;
  v80 = v8;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v81 = (__int64 *)(a2 + 24);
  v76[0] = v7;
  if ( v7 )
  {
    sub_1623A60((__int64)v76, (__int64)v7, 2);
    if ( v79 )
      sub_161E7C0((__int64)&v79, (__int64)v79);
    v79 = v76[0];
    if ( v76[0] )
      sub_1623210((__int64)v76, v76[0], (__int64)&v79);
  }
  v9 = **a1;
  v10 = sub_1649C60(*(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  v11 = sub_18165C0(v9, v10, a2, a3, a4, a5);
  v12 = **a1;
  v13 = sub_1649C60(*(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))));
  v14 = sub_18165C0(v12, v13, a2, a3, a4, a5);
  v75 = 257;
  v15 = sub_15A0680(**(_QWORD **)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))), 2, 0);
  v16 = *(_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  if ( *(_BYTE *)(v16 + 16) > 0x10u || *(_BYTE *)(v15 + 16) > 0x10u )
  {
    LOWORD(v77) = 257;
    v43 = sub_15FB440(15, (__int64 *)v16, v15, (__int64)v76, 0);
    v72 = v43;
    if ( v80 )
    {
      v44 = v81;
      sub_157E9D0(v80 + 40, v43);
      v45 = *v44;
      v46 = *(_QWORD *)(v72 + 24);
      *(_QWORD *)(v72 + 32) = v44;
      v45 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v72 + 24) = v45 | v46 & 7;
      *(_QWORD *)(v45 + 8) = v72 + 24;
      *v44 = *v44 & 7 | (v72 + 24);
    }
    sub_164B780(v72, v74);
    sub_12A86E0((__int64 *)&v79, v72);
  }
  else
  {
    v72 = sub_15A2C20((__int64 *)v16, v15, 0, 0, a3, a4, a5);
  }
  v17 = (__int64 **)sub_16471D0(*(_QWORD **)(**a1 + 168), 0);
  v75 = 257;
  v18 = v17;
  if ( v17 != *(__int64 ***)v11 )
  {
    if ( *(_BYTE *)(v11 + 16) > 0x10u )
    {
      LOWORD(v77) = 257;
      v68 = v17;
      v53 = sub_15FDBD0(47, v11, (__int64)v17, (__int64)v76, 0);
      v54 = v68;
      v11 = v53;
      if ( v80 )
      {
        v64 = v68;
        v69 = v81;
        sub_157E9D0(v80 + 40, v53);
        v54 = v64;
        v55 = *v69;
        v56 = *(_QWORD *)(v11 + 24) & 7LL;
        *(_QWORD *)(v11 + 32) = v69;
        v55 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v11 + 24) = v55 | v56;
        *(_QWORD *)(v55 + 8) = v11 + 24;
        *v69 = *v69 & 7 | (v11 + 24);
      }
      v70 = v54;
      sub_164B780(v11, v74);
      v18 = v70;
      if ( v79 )
      {
        v73 = v79;
        sub_1623A60((__int64)&v73, (__int64)v79, 2);
        v57 = *(_QWORD *)(v11 + 48);
        v58 = v11 + 48;
        v18 = v70;
        if ( v57 )
        {
          sub_161E7C0(v11 + 48, v57);
          v18 = v70;
          v58 = v11 + 48;
        }
        v59 = v73;
        *(_QWORD *)(v11 + 48) = v73;
        if ( v59 )
        {
          v71 = v18;
          sub_1623210((__int64)&v73, v59, v58);
          v18 = v71;
        }
      }
    }
    else
    {
      v66 = v17;
      v19 = sub_15A46C0(47, (__int64 ***)v11, v17, 0);
      v18 = v66;
      v11 = v19;
    }
  }
  v75 = 257;
  if ( v18 != *(__int64 ***)v14 )
  {
    if ( *(_BYTE *)(v14 + 16) > 0x10u )
    {
      LOWORD(v77) = 257;
      v47 = sub_15FDBD0(47, v14, (__int64)v18, (__int64)v76, 0);
      v14 = v47;
      if ( v80 )
      {
        v67 = v81;
        sub_157E9D0(v80 + 40, v47);
        v48 = *v67;
        v49 = *(_QWORD *)(v14 + 24) & 7LL;
        *(_QWORD *)(v14 + 32) = v67;
        v48 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v14 + 24) = v48 | v49;
        *(_QWORD *)(v48 + 8) = v14 + 24;
        *v67 = *v67 & 7 | (v14 + 24);
      }
      sub_164B780(v14, v74);
      if ( v79 )
      {
        v73 = v79;
        sub_1623A60((__int64)&v73, (__int64)v79, 2);
        v50 = *(_QWORD *)(v14 + 48);
        v51 = v14 + 48;
        if ( v50 )
        {
          sub_161E7C0(v14 + 48, v50);
          v51 = v14 + 48;
        }
        v52 = v73;
        *(_QWORD *)(v14 + 48) = v73;
        if ( v52 )
          sub_1623210((__int64)&v73, v52, v51);
      }
    }
    else
    {
      v14 = sub_15A46C0(47, (__int64 ***)v14, v18, 0);
    }
  }
  v75 = 257;
  v20 = *(_DWORD *)(a2 + 20);
  v77 = v72;
  v76[1] = (unsigned __int8 *)v14;
  v21 = 3LL - (v20 & 0xFFFFFFF);
  v22 = *(_QWORD *)(a2 - 24);
  v76[0] = (unsigned __int8 *)v11;
  v78 = *(_QWORD *)(a2 + 24 * v21);
  v23 = sub_1285290((__int64 *)&v79, *(_QWORD *)(*(_QWORD *)v22 + 24LL), v22, (int)v76, 4, (__int64)v74, 0);
  v24 = v23;
  if ( !byte_4FA9380 )
  {
    v76[0] = *(unsigned __int8 **)(v23 + 56);
    v31 = (__int64 *)sub_16498A0(v23);
    *(_QWORD *)(v24 + 56) = sub_1563C10((__int64 *)v76, v31, 1, 1);
    v32 = (__int64 *)sub_16498A0(v24);
    v33 = sub_155D330(v32, 2);
    LODWORD(v74[0]) = 0;
    v34 = v33;
    v76[0] = *(unsigned __int8 **)(v24 + 56);
    v35 = (__int64 *)sub_16498A0(v24);
    v36 = (unsigned __int8 *)sub_1563E10((__int64 *)v76, v35, (int *)v74, 1, v34);
    *(_QWORD *)(v24 + 56) = v36;
    v76[0] = v36;
    v37 = (__int64 *)sub_16498A0(v24);
    *(_QWORD *)(v24 + 56) = sub_1563C10((__int64 *)v76, v37, 2, 1);
    v38 = 2;
    v39 = (__int64 *)sub_16498A0(v24);
LABEL_23:
    v40 = sub_155D330(v39, v38);
    LODWORD(v74[0]) = 1;
    v41 = v40;
    v76[0] = *(unsigned __int8 **)(v24 + 56);
    v42 = (__int64 *)sub_16498A0(v24);
    result = (unsigned __int8 *)sub_1563E10((__int64 *)v76, v42, (int *)v74, 1, v41);
    v76[0] = result;
    *(_QWORD *)(v24 + 56) = result;
    goto LABEL_19;
  }
  v25 = (_QWORD *)(a2 + 56);
  v26 = 2 * sub_15603A0(v25, 0);
  v76[0] = *(unsigned __int8 **)(v24 + 56);
  v27 = (__int64 *)sub_16498A0(v24);
  *(_QWORD *)(v24 + 56) = sub_1563C10((__int64 *)v76, v27, 1, 1);
  if ( v26 )
  {
    v60 = (__int64 *)sub_16498A0(v24);
    v61 = sub_155D330(v60, v26);
    LODWORD(v74[0]) = 0;
    v62 = v61;
    v76[0] = *(unsigned __int8 **)(v24 + 56);
    v63 = (__int64 *)sub_16498A0(v24);
    v76[0] = (unsigned __int8 *)sub_1563E10((__int64 *)v76, v63, (int *)v74, 1, v62);
    *(unsigned __int8 **)(v24 + 56) = v76[0];
  }
  v28 = 2 * sub_15603A0(v25, 1);
  v76[0] = *(unsigned __int8 **)(v24 + 56);
  v29 = (__int64 *)sub_16498A0(v24);
  result = (unsigned __int8 *)sub_1563C10((__int64 *)v76, v29, 2, 1);
  *(_QWORD *)(v24 + 56) = result;
  if ( v28 )
  {
    v38 = v28;
    v39 = (__int64 *)sub_16498A0(v24);
    goto LABEL_23;
  }
LABEL_19:
  if ( v79 )
    return (unsigned __int8 *)sub_161E7C0((__int64)&v79, (__int64)v79);
  return result;
}
