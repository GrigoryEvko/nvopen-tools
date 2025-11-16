// Function: sub_1B919A0
// Address: 0x1b919a0
//
__int64 __fastcall sub_1B919A0(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        double a6,
        double a7,
        double a8)
{
  __int64 v9; // r14
  __int64 v10; // r13
  __int64 v11; // r12
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 *v15; // r15
  __int64 v16; // rax
  __int64 v17; // rcx
  unsigned __int64 v18; // rax
  _QWORD *v19; // r15
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rbx
  _QWORD *v28; // rax
  __int64 v29; // r15
  __int64 *v30; // rbx
  __int64 v31; // rax
  __int64 v32; // rcx
  _QWORD *v33; // rax
  __int64 v35; // rax
  unsigned __int64 *v36; // rbx
  __int64 v37; // rax
  unsigned __int64 v38; // rcx
  _QWORD *v39; // rax
  _QWORD **v40; // rax
  __int64 *v41; // rax
  __int64 v42; // rsi
  __int64 v43; // rsi
  __int64 v44; // rax
  __int64 v46; // [rsp+20h] [rbp-F0h]
  __int64 v47; // [rsp+28h] [rbp-E8h]
  _QWORD *v48; // [rsp+28h] [rbp-E8h]
  __int64 v50; // [rsp+30h] [rbp-E0h]
  __int64 *v52; // [rsp+38h] [rbp-D8h]
  __int64 v54; // [rsp+48h] [rbp-C8h]
  __int64 v55[2]; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v56; // [rsp+60h] [rbp-B0h]
  __int64 v57[2]; // [rsp+70h] [rbp-A0h] BYREF
  __int16 v58; // [rsp+80h] [rbp-90h]
  __int64 v59; // [rsp+90h] [rbp-80h] BYREF
  __int64 v60; // [rsp+98h] [rbp-78h]
  __int64 *v61; // [rsp+A0h] [rbp-70h]
  __int64 v62; // [rsp+A8h] [rbp-68h]
  __int64 v63; // [rsp+B0h] [rbp-60h]
  int v64; // [rsp+B8h] [rbp-58h]
  __int64 v65; // [rsp+C0h] [rbp-50h]
  __int64 v66; // [rsp+C8h] [rbp-48h]

  v9 = **(_QWORD **)(a2 + 32);
  v10 = sub_13FCB50(a2);
  if ( !v10 )
    v10 = v9;
  v11 = sub_157EE30(v9);
  if ( v11 )
    v11 -= 24;
  v59 = 0;
  v62 = sub_16498A0(v11);
  v61 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v60 = 0;
  sub_17050D0(&v59, v11);
  v47 = sub_1B8F7D0(*(_QWORD *)(a1 + 272));
  sub_1B91520(a1, &v59, v47);
  v12 = *a3;
  v55[0] = (__int64)"index";
  v56 = 259;
  v58 = 257;
  v13 = sub_1648B60(64);
  v14 = v13;
  if ( v13 )
  {
    v46 = v13;
    sub_15F1EA0(v13, v12, 53, 0, 0, 0);
    *(_DWORD *)(v14 + 56) = 2;
    sub_164B780(v14, v57);
    sub_1648880(v14, *(_DWORD *)(v14 + 56), 1);
  }
  else
  {
    v46 = 0;
  }
  if ( v60 )
  {
    v15 = v61;
    sub_157E9D0(v60 + 40, v14);
    v16 = *(_QWORD *)(v14 + 24);
    v17 = *v15;
    *(_QWORD *)(v14 + 32) = v15;
    v17 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v14 + 24) = v17 | v16 & 7;
    *(_QWORD *)(v17 + 8) = v14 + 24;
    *v15 = *v15 & 7 | (v14 + 24);
  }
  sub_164B780(v46, v55);
  sub_12A86E0(&v59, v14);
  v18 = sub_157EBA0(v10);
  sub_17050D0(&v59, v18);
  sub_1B91520(a1, &v59, v47);
  v55[0] = (__int64)"index.next";
  v56 = 259;
  if ( *(_BYTE *)(v14 + 16) > 0x10u || *(_BYTE *)(a5 + 16) > 0x10u )
  {
    v58 = 257;
    v35 = sub_15FB440(11, (__int64 *)v14, a5, (__int64)v57, 0);
    v19 = (_QWORD *)v35;
    if ( v60 )
    {
      v36 = (unsigned __int64 *)v61;
      sub_157E9D0(v60 + 40, v35);
      v37 = v19[3];
      v38 = *v36;
      v19[4] = v36;
      v38 &= 0xFFFFFFFFFFFFFFF8LL;
      v19[3] = v38 | v37 & 7;
      *(_QWORD *)(v38 + 8) = v19 + 3;
      *v36 = *v36 & 7 | (unsigned __int64)(v19 + 3);
    }
    sub_164B780((__int64)v19, v55);
    sub_12A86E0(&v59, (__int64)v19);
  }
  else
  {
    v19 = (_QWORD *)sub_15A2B30((__int64 *)v14, a5, 0, 0, a6, a7, a8);
  }
  v20 = sub_13FC520(a2);
  sub_1704F80(v14, (__int64)a3, v20, v21, v22, v23);
  sub_1704F80(v14, (__int64)v19, v10, v24, v25, v26);
  v56 = 257;
  if ( *((_BYTE *)v19 + 16) > 0x10u || *(_BYTE *)(a4 + 16) > 0x10u )
  {
    v58 = 257;
    v39 = sub_1648A60(56, 2u);
    v27 = (__int64)v39;
    if ( v39 )
    {
      v50 = (__int64)v39;
      v40 = (_QWORD **)*v19;
      if ( *(_BYTE *)(*v19 + 8LL) == 16 )
      {
        v48 = v40[4];
        v41 = (__int64 *)sub_1643320(*v40);
        v42 = (__int64)sub_16463B0(v41, (unsigned int)v48);
      }
      else
      {
        v42 = sub_1643320(*v40);
      }
      sub_15FEC10(v27, v42, 51, 32, (__int64)v19, a4, (__int64)v57, 0);
    }
    else
    {
      v50 = 0;
    }
    if ( v60 )
    {
      v52 = v61;
      sub_157E9D0(v60 + 40, v27);
      v43 = *v52;
      v44 = *(_QWORD *)(v27 + 24) & 7LL;
      *(_QWORD *)(v27 + 32) = v52;
      v43 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v27 + 24) = v43 | v44;
      *(_QWORD *)(v43 + 8) = v27 + 24;
      *v52 = *v52 & 7 | (v27 + 24);
    }
    sub_164B780(v50, v55);
    sub_12A86E0(&v59, v27);
  }
  else
  {
    v27 = sub_15A37B0(0x20u, v19, (_QWORD *)a4, 0);
  }
  v54 = sub_13FA090(a2);
  v58 = 257;
  v28 = sub_1648A60(56, 3u);
  v29 = (__int64)v28;
  if ( v28 )
    sub_15F83E0((__int64)v28, v54, v9, v27, 0);
  if ( v60 )
  {
    v30 = v61;
    sub_157E9D0(v60 + 40, v29);
    v31 = *(_QWORD *)(v29 + 24);
    v32 = *v30;
    *(_QWORD *)(v29 + 32) = v30;
    v32 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v29 + 24) = v32 | v31 & 7;
    *(_QWORD *)(v32 + 8) = v29 + 24;
    *v30 = *v30 & 7 | (v29 + 24);
  }
  sub_164B780(v29, v57);
  sub_12A86E0(&v59, v29);
  v33 = (_QWORD *)sub_157EBA0(v10);
  sub_15F20C0(v33);
  if ( v59 )
    sub_161E7C0((__int64)&v59, v59);
  return v14;
}
