// Function: sub_20CBD50
// Address: 0x20cbd50
//
unsigned __int64 __fastcall sub_20CBD50(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned int v11; // r13d
  __int64 v12; // rax
  __int64 *v13; // rsi
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // r12
  _QWORD *v17; // r12
  __int64 v18; // rax
  __int64 **v19; // r13
  __int64 v20; // r12
  double v21; // xmm4_8
  double v22; // xmm5_8
  unsigned __int64 result; // rax
  __int64 v24; // rax
  unsigned __int64 *v25; // r13
  __int64 v26; // rax
  unsigned __int64 v27; // rsi
  __int64 v28; // rsi
  unsigned __int8 *v29; // rsi
  __int64 v30; // rax
  __int64 *v31; // rbx
  __int64 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // rsi
  unsigned __int8 *v35; // rsi
  __int64 v36; // rax
  __int64 *v37; // rbx
  __int64 v38; // rax
  __int64 v39; // rcx
  __int64 v40; // rsi
  unsigned __int8 *v41; // rsi
  __int64 v42; // rsi
  __int64 v43; // rax
  __int64 *v44; // rbx
  __int64 v45; // rax
  __int64 v46; // rcx
  __int64 v47; // rsi
  __int64 v48; // rdx
  unsigned __int8 *v49; // rsi
  unsigned int v50; // [rsp+14h] [rbp-14Ch]
  __int64 *v51; // [rsp+18h] [rbp-148h] BYREF
  _QWORD *v52; // [rsp+20h] [rbp-140h] BYREF
  __int64 *v53; // [rsp+28h] [rbp-138h] BYREF
  _QWORD v54[4]; // [rsp+30h] [rbp-130h] BYREF
  __int64 v55[2]; // [rsp+50h] [rbp-110h] BYREF
  __int16 v56; // [rsp+60h] [rbp-100h]
  __int64 v57[2]; // [rsp+70h] [rbp-F0h] BYREF
  __int16 v58; // [rsp+80h] [rbp-E0h]
  _BYTE v59[16]; // [rsp+90h] [rbp-D0h] BYREF
  __int16 v60; // [rsp+A0h] [rbp-C0h]
  __int64 *v61[3]; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v62; // [rsp+C8h] [rbp-98h]
  __int64 *v63; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v64; // [rsp+E8h] [rbp-78h]
  __int64 *v65; // [rsp+F0h] [rbp-70h]
  __int64 v66; // [rsp+F8h] [rbp-68h]
  __int64 v67; // [rsp+100h] [rbp-60h]
  int v68; // [rsp+108h] [rbp-58h]
  __int64 v69; // [rsp+110h] [rbp-50h]
  __int64 v70; // [rsp+118h] [rbp-48h]

  v11 = *(unsigned __int16 *)(a2 + 18);
  v51 = (__int64 *)a2;
  v50 = (v11 >> 2) & 7;
  v12 = sub_16498A0(a2);
  v69 = 0;
  v70 = 0;
  v13 = *(__int64 **)(a2 + 48);
  v66 = v12;
  v68 = 0;
  v14 = *(_QWORD *)(a2 + 40);
  v63 = 0;
  v64 = v14;
  v67 = 0;
  v65 = (__int64 *)(a2 + 24);
  v61[0] = v13;
  if ( v13 )
  {
    sub_1623A60((__int64)v61, (__int64)v13, 2);
    if ( v63 )
      sub_161E7C0((__int64)&v63, (__int64)v63);
    v63 = v61[0];
    if ( v61[0] )
      sub_1623210((__int64)v61, (unsigned __int8 *)v61[0], (__int64)&v63);
  }
  sub_20CB200(
    v61,
    (__int64 *)&v63,
    (__int64)v51,
    *v51,
    (__int64 ***)*(v51 - 6),
    *(_DWORD *)(*(_QWORD *)(a1 + 160) + 104LL) >> 3,
    *(double *)a3.m128_u64,
    a4,
    a5);
  v15 = v62;
  v57[0] = (__int64)"ValOperand_Shifted";
  v56 = 257;
  v58 = 259;
  v16 = *(v51 - 3);
  if ( v61[0] != *(__int64 **)v16 )
  {
    if ( *(_BYTE *)(v16 + 16) > 0x10u )
    {
      v42 = *(v51 - 3);
      v60 = 257;
      v43 = sub_15FDBD0(37, v42, (__int64)v61[0], (__int64)v59, 0);
      v16 = v43;
      if ( v64 )
      {
        v44 = v65;
        sub_157E9D0(v64 + 40, v43);
        v45 = *(_QWORD *)(v16 + 24);
        v46 = *v44;
        *(_QWORD *)(v16 + 32) = v44;
        v46 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v16 + 24) = v46 | v45 & 7;
        *(_QWORD *)(v46 + 8) = v16 + 24;
        *v44 = *v44 & 7 | (v16 + 24);
      }
      sub_164B780(v16, v55);
      if ( v63 )
      {
        v54[0] = v63;
        sub_1623A60((__int64)v54, (__int64)v63, 2);
        v47 = *(_QWORD *)(v16 + 48);
        v48 = v16 + 48;
        if ( v47 )
        {
          sub_161E7C0(v16 + 48, v47);
          v48 = v16 + 48;
        }
        v49 = (unsigned __int8 *)v54[0];
        *(_QWORD *)(v16 + 48) = v54[0];
        if ( v49 )
          sub_1623210((__int64)v54, v49, v48);
      }
    }
    else
    {
      v16 = sub_15A46C0(37, (__int64 ***)*(v51 - 3), (__int64 **)v61[0], 0);
    }
  }
  if ( *(_BYTE *)(v16 + 16) > 0x10u || *(_BYTE *)(v15 + 16) > 0x10u )
  {
    v60 = 257;
    v24 = sub_15FB440(23, (__int64 *)v16, v15, (__int64)v59, 0);
    v17 = (_QWORD *)v24;
    if ( v64 )
    {
      v25 = (unsigned __int64 *)v65;
      sub_157E9D0(v64 + 40, v24);
      v26 = v17[3];
      v27 = *v25;
      v17[4] = v25;
      v27 &= 0xFFFFFFFFFFFFFFF8LL;
      v17[3] = v27 | v26 & 7;
      *(_QWORD *)(v27 + 8) = v17 + 3;
      *v25 = *v25 & 7 | (unsigned __int64)(v17 + 3);
    }
    sub_164B780((__int64)v17, v57);
    if ( v63 )
    {
      v54[0] = v63;
      sub_1623A60((__int64)v54, (__int64)v63, 2);
      v28 = v17[6];
      if ( v28 )
        sub_161E7C0((__int64)(v17 + 6), v28);
      v29 = (unsigned __int8 *)v54[0];
      v17[6] = v54[0];
      if ( v29 )
        sub_1623210((__int64)v54, v29, (__int64)(v17 + 6));
    }
  }
  else
  {
    v17 = (_QWORD *)sub_15A2D50((__int64 *)v16, v15, 0, 0, *(double *)a3.m128_u64, a4, a5);
  }
  v54[0] = &v51;
  v54[1] = &v52;
  v52 = v17;
  v54[2] = v61;
  v18 = sub_20C9DC0(
          (__int64 *)&v63,
          (__int64)v61[0],
          (__int64)v61[2],
          v50,
          (__int64 (__fastcall *)(__int64, __int64 *, __int64))sub_20CD370,
          (__int64)v54,
          (void (__fastcall *)(__int64, __int64 *, __int64, __int64, __int64, _QWORD, unsigned __int8 **, __int64 *))sub_20C9610,
          (__int64)sub_20CAC10);
  v58 = 257;
  v19 = (__int64 **)v61[1];
  v56 = 257;
  if ( *(_BYTE *)(v18 + 16) > 0x10u || *(_BYTE *)(v62 + 16) > 0x10u )
  {
    v60 = 257;
    v30 = sub_15FB440(24, (__int64 *)v18, v62, (__int64)v59, 0);
    v20 = v30;
    if ( v64 )
    {
      v31 = v65;
      sub_157E9D0(v64 + 40, v30);
      v32 = *(_QWORD *)(v20 + 24);
      v33 = *v31;
      *(_QWORD *)(v20 + 32) = v31;
      v33 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v20 + 24) = v33 | v32 & 7;
      *(_QWORD *)(v33 + 8) = v20 + 24;
      *v31 = *v31 & 7 | (v20 + 24);
    }
    sub_164B780(v20, v55);
    if ( v63 )
    {
      v53 = v63;
      sub_1623A60((__int64)&v53, (__int64)v63, 2);
      v34 = *(_QWORD *)(v20 + 48);
      if ( v34 )
        sub_161E7C0(v20 + 48, v34);
      v35 = (unsigned __int8 *)v53;
      *(_QWORD *)(v20 + 48) = v53;
      if ( v35 )
        sub_1623210((__int64)&v53, v35, v20 + 48);
    }
  }
  else
  {
    v20 = sub_15A2D80((__int64 *)v18, v62, 0, *(double *)a3.m128_u64, a4, a5);
  }
  if ( v19 != *(__int64 ***)v20 )
  {
    if ( *(_BYTE *)(v20 + 16) > 0x10u )
    {
      v60 = 257;
      v36 = sub_15FDBD0(36, v20, (__int64)v19, (__int64)v59, 0);
      v20 = v36;
      if ( v64 )
      {
        v37 = v65;
        sub_157E9D0(v64 + 40, v36);
        v38 = *(_QWORD *)(v20 + 24);
        v39 = *v37;
        *(_QWORD *)(v20 + 32) = v37;
        v39 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v20 + 24) = v39 | v38 & 7;
        *(_QWORD *)(v39 + 8) = v20 + 24;
        *v37 = *v37 & 7 | (v20 + 24);
      }
      sub_164B780(v20, v57);
      if ( v63 )
      {
        v53 = v63;
        sub_1623A60((__int64)&v53, (__int64)v63, 2);
        v40 = *(_QWORD *)(v20 + 48);
        if ( v40 )
          sub_161E7C0(v20 + 48, v40);
        v41 = (unsigned __int8 *)v53;
        *(_QWORD *)(v20 + 48) = v53;
        if ( v41 )
          sub_1623210((__int64)&v53, v41, v20 + 48);
      }
    }
    else
    {
      v20 = sub_15A46C0(36, (__int64 ***)v20, v19, 0);
    }
  }
  sub_164D160((__int64)v51, v20, a3, a4, a5, a6, v21, v22, a9, a10);
  result = sub_15F20C0(v51);
  if ( v63 )
    return sub_161E7C0((__int64)&v63, (__int64)v63);
  return result;
}
