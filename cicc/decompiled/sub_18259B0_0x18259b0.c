// Function: sub_18259B0
// Address: 0x18259b0
//
__int64 __fastcall sub_18259B0(
        _QWORD *a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned int a4,
        double a5,
        double a6,
        double a7)
{
  _QWORD *v10; // r12
  _QWORD *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 result; // rax
  _QWORD *v17; // rax
  unsigned __int8 *v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r13
  __int64 *v23; // rax
  __int64 *v24; // rax
  __int64 v25; // r13
  _QWORD *v26; // rax
  __int64 v27; // r12
  __int64 *v28; // rbx
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rsi
  unsigned __int8 *v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rax
  _QWORD *v35; // r12
  _QWORD *v36; // rax
  _QWORD *v37; // rbx
  unsigned __int64 *v38; // r12
  __int64 v39; // rax
  unsigned __int64 v40; // rcx
  __int64 v41; // rsi
  unsigned __int8 *v42; // rsi
  __int64 v43; // rax
  _QWORD *v44; // rax
  _QWORD *v45; // rdx
  _BOOL8 v46; // rdi
  __int64 v47; // rax
  unsigned __int64 *v48; // rbx
  __int64 v49; // rax
  unsigned __int64 v50; // rcx
  __int64 v51; // rsi
  unsigned __int8 *v52; // rsi
  _QWORD *v53; // [rsp+0h] [rbp-F0h]
  _QWORD *v54; // [rsp+8h] [rbp-E8h]
  _QWORD *v55; // [rsp+8h] [rbp-E8h]
  unsigned __int8 *v56; // [rsp+18h] [rbp-D8h] BYREF
  __int64 *v57[2]; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v58[2]; // [rsp+30h] [rbp-C0h] BYREF
  __int16 v59; // [rsp+40h] [rbp-B0h]
  unsigned __int8 *v60[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v61; // [rsp+60h] [rbp-90h]
  unsigned __int8 *v62; // [rsp+70h] [rbp-80h] BYREF
  __int64 v63; // [rsp+78h] [rbp-78h]
  __int64 *v64; // [rsp+80h] [rbp-70h]
  _QWORD *v65; // [rsp+88h] [rbp-68h]
  __int64 v66; // [rsp+90h] [rbp-60h]
  int v67; // [rsp+98h] [rbp-58h]
  __int64 v68; // [rsp+A0h] [rbp-50h]
  __int64 v69; // [rsp+A8h] [rbp-48h]

  v10 = a1 + 50;
  v12 = (_QWORD *)a1[51];
  if ( !v12 )
    goto LABEL_38;
  do
  {
    while ( 1 )
    {
      v13 = v12[2];
      v14 = v12[3];
      if ( v12[4] >= a3 )
        break;
      v12 = (_QWORD *)v12[3];
      if ( !v14 )
        goto LABEL_6;
    }
    v10 = v12;
    v12 = (_QWORD *)v12[2];
  }
  while ( v13 );
LABEL_6:
  if ( v10 == a1 + 50 || v10[4] > a3 )
  {
LABEL_38:
    v54 = v10;
    v53 = a1 + 50;
    v43 = sub_22077B0(48);
    *(_QWORD *)(v43 + 32) = a3;
    v10 = (_QWORD *)v43;
    *(_QWORD *)(v43 + 40) = 0;
    v44 = sub_18258B0(a1 + 49, v54, (unsigned __int64 *)(v43 + 32));
    if ( v45 )
    {
      v46 = v53 == v45 || v44 || a3 < v45[4];
      sub_220F040(v46, v10, v45, v53);
      ++a1[54];
    }
    else
    {
      v55 = v44;
      j_j___libc_free_0(v10, 48);
      v10 = v55;
    }
  }
  v15 = v10[5];
  result = 0;
  if ( v15 )
  {
    v17 = (_QWORD *)sub_16498A0(a2);
    v18 = *(unsigned __int8 **)(a2 + 48);
    v62 = 0;
    v65 = v17;
    v19 = *(_QWORD *)(a2 + 40);
    v66 = 0;
    v63 = v19;
    v67 = 0;
    v68 = 0;
    v69 = 0;
    v64 = (__int64 *)(a2 + 24);
    v60[0] = v18;
    if ( v18 )
    {
      sub_1623A60((__int64)v60, (__int64)v18, 2);
      if ( v62 )
        sub_161E7C0((__int64)&v62, (__int64)v62);
      v62 = v60[0];
      if ( v60[0] )
        sub_1623210((__int64)v60, v60[0], (__int64)&v62);
    }
    v20 = sub_1643350(v65);
    v57[0] = (__int64 *)sub_159C470(v20, 0, 0);
    v21 = sub_1643350(v65);
    v57[1] = (__int64 *)sub_159C470(v21, a4, 0);
    v22 = (unsigned int)(*(_DWORD *)(a3 + 12) + 1);
    v23 = (__int64 *)sub_1643360(v65);
    v24 = sub_1645D80(v23, v22);
    BYTE4(v60[0]) = 0;
    v25 = sub_15A2E80((__int64)v24, v15, v57, 2u, 0, (__int64)v60, 0);
    v61 = 257;
    v26 = sub_1648A60(64, 1u);
    v27 = (__int64)v26;
    if ( v26 )
      sub_15F9210((__int64)v26, *(_QWORD *)(*(_QWORD *)v25 + 24LL), v25, 0, 0, 0);
    if ( v63 )
    {
      v28 = v64;
      sub_157E9D0(v63 + 40, v27);
      v29 = *(_QWORD *)(v27 + 24);
      v30 = *v28;
      *(_QWORD *)(v27 + 32) = v28;
      v30 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v27 + 24) = v30 | v29 & 7;
      *(_QWORD *)(v30 + 8) = v27 + 24;
      *v28 = *v28 & 7 | (v27 + 24);
    }
    sub_164B780(v27, (__int64 *)v60);
    if ( v62 )
    {
      v58[0] = (__int64)v62;
      sub_1623A60((__int64)v58, (__int64)v62, 2);
      v31 = *(_QWORD *)(v27 + 48);
      if ( v31 )
        sub_161E7C0(v27 + 48, v31);
      v32 = (unsigned __int8 *)v58[0];
      *(_QWORD *)(v27 + 48) = v58[0];
      if ( v32 )
        sub_1623210((__int64)v58, v32, v27 + 48);
    }
    v59 = 257;
    v33 = sub_1643360(v65);
    v34 = sub_159C470(v33, 1, 0);
    if ( *(_BYTE *)(v27 + 16) > 0x10u || *(_BYTE *)(v34 + 16) > 0x10u )
    {
      v61 = 257;
      v47 = sub_15FB440(11, (__int64 *)v27, v34, (__int64)v60, 0);
      v35 = (_QWORD *)v47;
      if ( v63 )
      {
        v48 = (unsigned __int64 *)v64;
        sub_157E9D0(v63 + 40, v47);
        v49 = v35[3];
        v50 = *v48;
        v35[4] = v48;
        v50 &= 0xFFFFFFFFFFFFFFF8LL;
        v35[3] = v50 | v49 & 7;
        *(_QWORD *)(v50 + 8) = v35 + 3;
        *v48 = *v48 & 7 | (unsigned __int64)(v35 + 3);
      }
      sub_164B780((__int64)v35, v58);
      if ( v62 )
      {
        v56 = v62;
        sub_1623A60((__int64)&v56, (__int64)v62, 2);
        v51 = v35[6];
        if ( v51 )
          sub_161E7C0((__int64)(v35 + 6), v51);
        v52 = v56;
        v35[6] = v56;
        if ( v52 )
          sub_1623210((__int64)&v56, v52, (__int64)(v35 + 6));
      }
    }
    else
    {
      v35 = (_QWORD *)sub_15A2B30((__int64 *)v27, v34, 0, 0, a5, a6, a7);
    }
    v61 = 257;
    v36 = sub_1648A60(64, 2u);
    v37 = v36;
    if ( v36 )
      sub_15F9650((__int64)v36, (__int64)v35, v25, 0, 0);
    if ( v63 )
    {
      v38 = (unsigned __int64 *)v64;
      sub_157E9D0(v63 + 40, (__int64)v37);
      v39 = v37[3];
      v40 = *v38;
      v37[4] = v38;
      v40 &= 0xFFFFFFFFFFFFFFF8LL;
      v37[3] = v40 | v39 & 7;
      *(_QWORD *)(v40 + 8) = v37 + 3;
      *v38 = *v38 & 7 | (unsigned __int64)(v37 + 3);
    }
    sub_164B780((__int64)v37, (__int64 *)v60);
    result = 1;
    if ( v62 )
    {
      v56 = v62;
      sub_1623A60((__int64)&v56, (__int64)v62, 2);
      v41 = v37[6];
      if ( v41 )
        sub_161E7C0((__int64)(v37 + 6), v41);
      v42 = v56;
      v37[6] = v56;
      if ( v42 )
        sub_1623210((__int64)&v56, v42, (__int64)(v37 + 6));
      if ( v62 )
      {
        sub_161E7C0((__int64)&v62, (__int64)v62);
        return 1;
      }
      else
      {
        return 1;
      }
    }
  }
  return result;
}
