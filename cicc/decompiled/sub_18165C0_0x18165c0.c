// Function: sub_18165C0
// Address: 0x18165c0
//
__int64 __fastcall sub_18165C0(__int64 a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  __int64 v8; // rax
  unsigned __int8 *v9; // rsi
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 **v12; // r9
  __int64 **v13; // r14
  __int64 v14; // r15
  __int64 **v15; // rdx
  __int64 v16; // rax
  unsigned __int8 v17; // al
  __int64 v18; // r12
  unsigned int v20; // ebx
  __int64 v21; // rax
  __int64 *v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 *v26; // rbx
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // r14
  __int64 v30; // r15
  _QWORD *v31; // rax
  __int64 *v32; // r14
  __int64 v33; // rax
  __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // rsi
  __int64 v39; // rdx
  unsigned __int8 *v40; // rsi
  __int64 v41; // rax
  __int64 *v42; // rbx
  __int64 v43; // rax
  __int64 v44; // rcx
  __int64 v45; // rsi
  __int64 v46; // rdx
  unsigned __int8 *v47; // rsi
  __int64 v48; // rax
  __int64 *v49; // rbx
  __int64 v50; // rax
  __int64 v51; // rcx
  __int64 v52; // rsi
  unsigned __int8 *v53; // rsi
  __int64 *v54; // [rsp+8h] [rbp-158h]
  unsigned __int8 *v55; // [rsp+18h] [rbp-148h] BYREF
  __int64 v56[2]; // [rsp+20h] [rbp-140h] BYREF
  __int16 v57; // [rsp+30h] [rbp-130h]
  __int64 v58; // [rsp+40h] [rbp-120h] BYREF
  __int16 v59; // [rsp+50h] [rbp-110h]
  __int64 v60; // [rsp+60h] [rbp-100h] BYREF
  __int16 v61; // [rsp+70h] [rbp-F0h]
  __int64 v62; // [rsp+80h] [rbp-E0h] BYREF
  __int16 v63; // [rsp+90h] [rbp-D0h]
  __int64 v64; // [rsp+A0h] [rbp-C0h] BYREF
  __int16 v65; // [rsp+B0h] [rbp-B0h]
  unsigned __int8 *v66[2]; // [rsp+C0h] [rbp-A0h] BYREF
  __int16 v67; // [rsp+D0h] [rbp-90h]
  unsigned __int8 *v68; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v69; // [rsp+E8h] [rbp-78h]
  __int64 *v70; // [rsp+F0h] [rbp-70h]
  __int64 v71; // [rsp+F8h] [rbp-68h]
  __int64 v72; // [rsp+100h] [rbp-60h]
  int v73; // [rsp+108h] [rbp-58h]
  __int64 v74; // [rsp+110h] [rbp-50h]
  __int64 v75; // [rsp+118h] [rbp-48h]

  v8 = sub_16498A0(a3);
  v9 = *(unsigned __int8 **)(a3 + 48);
  v68 = 0;
  v71 = v8;
  v10 = *(_QWORD *)(a3 + 40);
  v72 = 0;
  v69 = v10;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v70 = (__int64 *)(a3 + 24);
  v66[0] = v9;
  if ( v9 )
  {
    sub_1623A60((__int64)v66, (__int64)v9, 2);
    if ( v68 )
      sub_161E7C0((__int64)&v68, (__int64)v68);
    v68 = v66[0];
    if ( v66[0] )
      sub_1623210((__int64)v66, v66[0], (__int64)&v68);
  }
  if ( *(_BYTE *)(a1 + 528) )
  {
    v29 = *(_QWORD *)(a1 + 192);
    v67 = 257;
    v30 = *(_QWORD *)(a1 + 272);
    v31 = sub_1648A60(64, 1u);
    v11 = (__int64)v31;
    if ( v31 )
      sub_15F9210((__int64)v31, v29, v30, 0, 0, 0);
    if ( v69 )
    {
      v32 = v70;
      sub_157E9D0(v69 + 40, v11);
      v33 = *(_QWORD *)(v11 + 24);
      v34 = *v32;
      *(_QWORD *)(v11 + 32) = v32;
      v34 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v11 + 24) = v34 | v33 & 7;
      *(_QWORD *)(v34 + 8) = v11 + 24;
      *v32 = *v32 & 7 | (v11 + 24);
    }
    sub_164B780(v11, (__int64 *)v66);
    sub_12A86E0((__int64 *)&v68, v11);
  }
  else
  {
    v11 = *(_QWORD *)(a1 + 208);
  }
  v65 = 257;
  v12 = *(__int64 ***)(a1 + 192);
  v63 = 257;
  v13 = *(__int64 ***)(a1 + 184);
  v61 = 257;
  v14 = *(_QWORD *)(a1 + 216);
  v59 = 257;
  v15 = *(__int64 ***)v11;
  if ( v12 != *(__int64 ***)v11 )
  {
    if ( *(_BYTE *)(v11 + 16) > 0x10u )
    {
      v67 = 257;
      v35 = sub_15FDBD0(45, v11, (__int64)v12, (__int64)v66, 0);
      v11 = v35;
      if ( v69 )
      {
        v54 = v70;
        sub_157E9D0(v69 + 40, v35);
        v36 = *v54;
        v37 = *(_QWORD *)(v11 + 24) & 7LL;
        *(_QWORD *)(v11 + 32) = v54;
        v36 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v11 + 24) = v36 | v37;
        *(_QWORD *)(v36 + 8) = v11 + 24;
        *v54 = *v54 & 7 | (v11 + 24);
      }
      sub_164B780(v11, &v58);
      if ( v68 )
      {
        v56[0] = (__int64)v68;
        sub_1623A60((__int64)v56, (__int64)v68, 2);
        v38 = *(_QWORD *)(v11 + 48);
        v39 = v11 + 48;
        if ( v38 )
        {
          sub_161E7C0(v11 + 48, v38);
          v39 = v11 + 48;
        }
        v40 = (unsigned __int8 *)v56[0];
        *(_QWORD *)(v11 + 48) = v56[0];
        if ( v40 )
          sub_1623210((__int64)v56, v40, v39);
      }
      v15 = *(__int64 ***)(a1 + 192);
    }
    else
    {
      v16 = sub_15A46C0(45, (__int64 ***)v11, v12, 0);
      v15 = *(__int64 ***)(a1 + 192);
      v11 = v16;
    }
  }
  v57 = 257;
  if ( *(__int64 ***)a2 != v15 )
  {
    if ( *(_BYTE *)(a2 + 16) > 0x10u )
    {
      v67 = 257;
      v41 = sub_15FDBD0(45, a2, (__int64)v15, (__int64)v66, 0);
      a2 = v41;
      if ( v69 )
      {
        v42 = v70;
        sub_157E9D0(v69 + 40, v41);
        v43 = *(_QWORD *)(a2 + 24);
        v44 = *v42;
        *(_QWORD *)(a2 + 32) = v42;
        v44 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(a2 + 24) = v44 | v43 & 7;
        *(_QWORD *)(v44 + 8) = a2 + 24;
        *v42 = *v42 & 7 | (a2 + 24);
      }
      sub_164B780(a2, v56);
      if ( v68 )
      {
        v55 = v68;
        sub_1623A60((__int64)&v55, (__int64)v68, 2);
        v45 = *(_QWORD *)(a2 + 48);
        v46 = a2 + 48;
        if ( v45 )
        {
          sub_161E7C0(a2 + 48, v45);
          v46 = a2 + 48;
        }
        v47 = v55;
        *(_QWORD *)(a2 + 48) = v55;
        if ( v47 )
          sub_1623210((__int64)&v55, v47, v46);
      }
    }
    else
    {
      a2 = sub_15A46C0(45, (__int64 ***)a2, v15, 0);
    }
  }
  v17 = *(_BYTE *)(v11 + 16);
  if ( v17 > 0x10u )
    goto LABEL_34;
  if ( v17 == 13 )
  {
    v20 = *(_DWORD *)(v11 + 32);
    if ( v20 > 0x40 )
    {
      if ( v20 == (unsigned int)sub_16A58F0(v11 + 24) )
        goto LABEL_18;
      if ( *(_BYTE *)(a2 + 16) <= 0x10u )
        goto LABEL_17;
      goto LABEL_34;
    }
    if ( *(_QWORD *)(v11 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v20) )
      goto LABEL_18;
  }
  if ( *(_BYTE *)(a2 + 16) <= 0x10u )
  {
LABEL_17:
    a2 = sub_15A2CF0((__int64 *)a2, v11, a4, a5, a6);
    goto LABEL_18;
  }
LABEL_34:
  v67 = 257;
  v25 = sub_15FB440(26, (__int64 *)a2, v11, (__int64)v66, 0);
  a2 = v25;
  if ( v69 )
  {
    v26 = v70;
    sub_157E9D0(v69 + 40, v25);
    v27 = *(_QWORD *)(a2 + 24);
    v28 = *v26;
    *(_QWORD *)(a2 + 32) = v26;
    v28 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a2 + 24) = v28 | v27 & 7;
    *(_QWORD *)(v28 + 8) = a2 + 24;
    *v26 = *v26 & 7 | (a2 + 24);
  }
  sub_164B780(a2, &v60);
  sub_12A86E0((__int64 *)&v68, a2);
LABEL_18:
  if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(v14 + 16) > 0x10u )
  {
    v67 = 257;
    v21 = sub_15FB440(15, (__int64 *)a2, v14, (__int64)v66, 0);
    v18 = v21;
    if ( v69 )
    {
      v22 = v70;
      sub_157E9D0(v69 + 40, v21);
      v23 = *(_QWORD *)(v18 + 24);
      v24 = *v22;
      *(_QWORD *)(v18 + 32) = v22;
      v24 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v18 + 24) = v24 | v23 & 7;
      *(_QWORD *)(v24 + 8) = v18 + 24;
      *v22 = *v22 & 7 | (v18 + 24);
    }
    sub_164B780(v18, &v62);
    sub_12A86E0((__int64 *)&v68, v18);
  }
  else
  {
    v18 = sub_15A2C20((__int64 *)a2, v14, 0, 0, a4, a5, a6);
  }
  if ( v13 == *(__int64 ***)v18 )
    goto LABEL_24;
  if ( *(_BYTE *)(v18 + 16) <= 0x10u )
  {
    v18 = sub_15A46C0(46, (__int64 ***)v18, v13, 0);
LABEL_24:
    if ( v68 )
      sub_161E7C0((__int64)&v68, (__int64)v68);
    return v18;
  }
  v67 = 257;
  v48 = sub_15FDBD0(46, v18, (__int64)v13, (__int64)v66, 0);
  v18 = v48;
  if ( v69 )
  {
    v49 = v70;
    sub_157E9D0(v69 + 40, v48);
    v50 = *(_QWORD *)(v18 + 24);
    v51 = *v49;
    *(_QWORD *)(v18 + 32) = v49;
    v51 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v18 + 24) = v51 | v50 & 7;
    *(_QWORD *)(v51 + 8) = v18 + 24;
    *v49 = *v49 & 7 | (v18 + 24);
  }
  sub_164B780(v18, &v64);
  if ( v68 )
  {
    v55 = v68;
    sub_1623A60((__int64)&v55, (__int64)v68, 2);
    v52 = *(_QWORD *)(v18 + 48);
    if ( v52 )
      sub_161E7C0(v18 + 48, v52);
    v53 = v55;
    *(_QWORD *)(v18 + 48) = v55;
    if ( v53 )
      sub_1623210((__int64)&v55, v53, v18 + 48);
    goto LABEL_24;
  }
  return v18;
}
