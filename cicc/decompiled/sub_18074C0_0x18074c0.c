// Function: sub_18074C0
// Address: 0x18074c0
//
unsigned __int64 __fastcall sub_18074C0(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        unsigned __int8 a6,
        double a7,
        double a8,
        double a9,
        char a10,
        unsigned int a11)
{
  _QWORD *v14; // rax
  unsigned __int8 *v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // r10
  unsigned __int8 v20; // bl
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  unsigned __int64 result; // rax
  __int64 v25; // rdx
  __int64 v26; // rdi
  __int64 **v27; // r11
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // r15
  __int64 v31; // rax
  __int64 v32; // r10
  __int64 *v33; // rbx
  __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rdx
  unsigned __int8 *v38; // rsi
  __int64 v39; // [rsp+10h] [rbp-110h]
  __int64 v40; // [rsp+18h] [rbp-108h]
  __int64 v41; // [rsp+18h] [rbp-108h]
  __int64 v42; // [rsp+18h] [rbp-108h]
  __int64 v43; // [rsp+18h] [rbp-108h]
  unsigned int v44; // [rsp+24h] [rbp-FCh]
  __int64 v46; // [rsp+30h] [rbp-F0h]
  __int64 v48; // [rsp+38h] [rbp-E8h]
  __int64 v49; // [rsp+40h] [rbp-E0h]
  unsigned __int8 *v51; // [rsp+58h] [rbp-C8h] BYREF
  __int64 v52; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v53; // [rsp+68h] [rbp-B8h]
  __int64 v54; // [rsp+70h] [rbp-B0h]
  unsigned __int8 *v55[2]; // [rsp+80h] [rbp-A0h] BYREF
  __int16 v56; // [rsp+90h] [rbp-90h]
  unsigned __int8 *v57; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v58; // [rsp+A8h] [rbp-78h]
  __int64 *v59; // [rsp+B0h] [rbp-70h]
  _QWORD *v60; // [rsp+B8h] [rbp-68h]
  __int64 v61; // [rsp+C0h] [rbp-60h]
  int v62; // [rsp+C8h] [rbp-58h]
  __int64 v63; // [rsp+D0h] [rbp-50h]
  __int64 v64; // [rsp+D8h] [rbp-48h]

  v14 = (_QWORD *)sub_16498A0(a3);
  v15 = *(unsigned __int8 **)(a3 + 48);
  v57 = 0;
  v60 = v14;
  v16 = *(_QWORD *)(a3 + 40);
  v61 = 0;
  v58 = v16;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v59 = (__int64 *)(a3 + 24);
  v55[0] = v15;
  if ( v15 )
  {
    sub_1623A60((__int64)v55, (__int64)v15, 2);
    if ( v57 )
      sub_161E7C0((__int64)&v57, (__int64)v57);
    v57 = v55[0];
    if ( v55[0] )
      sub_1623210((__int64)v55, v55[0], (__int64)&v57);
  }
  v44 = a5 >> 3;
  v17 = sub_15A0680(*(_QWORD *)(a1 + 232), a5 >> 3, 0);
  v18 = *(_QWORD *)(a1 + 232);
  v49 = v17;
  LOWORD(v54) = 257;
  if ( v18 == *(_QWORD *)a4 )
  {
    v19 = a4;
  }
  else if ( *(_BYTE *)(a4 + 16) > 0x10u )
  {
    v56 = 257;
    v31 = sub_15FDFF0(a4, v18, (__int64)v55, 0);
    v32 = v31;
    if ( v58 )
    {
      v33 = v59;
      v40 = v31;
      sub_157E9D0(v58 + 40, v31);
      v32 = v40;
      v34 = *v33;
      v35 = *(_QWORD *)(v40 + 24);
      *(_QWORD *)(v40 + 32) = v33;
      v34 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v40 + 24) = v34 | v35 & 7;
      *(_QWORD *)(v34 + 8) = v40 + 24;
      *v33 = *v33 & 7 | (v40 + 24);
    }
    v41 = v32;
    sub_164B780(v32, &v52);
    v19 = v41;
    if ( v57 )
    {
      v51 = v57;
      sub_1623A60((__int64)&v51, (__int64)v57, 2);
      v19 = v41;
      v36 = *(_QWORD *)(v41 + 48);
      v37 = v41 + 48;
      if ( v36 )
      {
        v39 = v41;
        v42 = v41 + 48;
        sub_161E7C0(v42, v36);
        v19 = v39;
        v37 = v42;
      }
      v38 = v51;
      *(_QWORD *)(v19 + 48) = v51;
      if ( v38 )
      {
        v43 = v19;
        sub_1623210((__int64)&v51, v38, v37);
        v19 = v43;
      }
    }
  }
  else
  {
    v19 = sub_15A4A70((__int64 ***)a4, v18);
  }
  v20 = a6;
  if ( a10 )
  {
    if ( a11 )
    {
      v52 = v19;
      v56 = 257;
      v53 = v49;
      v21 = sub_1643350(v60);
      v22 = sub_159C470(v21, a11, 0);
      v23 = *(_QWORD *)(a1 + 16LL * a6 + 664);
      v54 = v22;
      result = sub_1285290((__int64 *)&v57, *(_QWORD *)(v23 + 24), v23, (int)&v52, 3, (__int64)v55, 0);
    }
    else
    {
      v25 = *(_QWORD *)(16LL * a6 + a1 + 656);
      v52 = v19;
      v56 = 257;
      v53 = v49;
      result = sub_1285290((__int64 *)&v57, *(_QWORD *)(v25 + 24), v25, (int)&v52, 2, (__int64)v55, 0);
    }
  }
  else
  {
    v26 = *(_QWORD *)(a1 + 232);
    v56 = 257;
    v27 = *(__int64 ***)a4;
    v46 = v19;
    LOWORD(v54) = 257;
    v48 = (__int64)v27;
    v28 = sub_15A0680(v26, v44 - 1, 0);
    v29 = sub_12899C0((__int64 *)&v57, v46, v28, (__int64)&v52, 0, 0);
    v30 = sub_12AA3B0((__int64 *)&v57, 0x2Eu, v29, v48, (__int64)v55);
    sub_1806810(a1, a2, a3, a4, 8u, v20, a7, a8, a9, v49, 0, a11);
    result = sub_1806810(a1, a2, a3, v30, 8u, v20, a7, a8, a9, v49, 0, a11);
  }
  if ( v57 )
    return sub_161E7C0((__int64)&v57, (__int64)v57);
  return result;
}
