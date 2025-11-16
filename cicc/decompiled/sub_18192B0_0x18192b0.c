// Function: sub_18192B0
// Address: 0x18192b0
//
__int64 __fastcall sub_18192B0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, char a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rsi
  _QWORD *v14; // rax
  __int64 v15; // r14
  _BYTE *v16; // rsi
  int v17; // edx
  __int64 v18; // r15
  __int64 v19; // rbx
  _QWORD *v20; // rax
  _QWORD *v21; // r11
  __int64 v22; // r13
  _QWORD *v23; // rdi
  _QWORD *v25; // rdi
  __m128i *v26; // rax
  _QWORD *v27; // rax
  char *v28; // rax
  signed __int64 v29; // rdx
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 v32; // rdi
  __int64 *v33; // rax
  __int64 v34; // r11
  _QWORD *v35; // rax
  __int64 v36; // r13
  _QWORD *v37; // rdi
  __int64 v38; // [rsp+0h] [rbp-120h]
  __int64 v40; // [rsp+8h] [rbp-118h]
  __int64 v41; // [rsp+10h] [rbp-110h]
  __int64 *v42; // [rsp+10h] [rbp-110h]
  int v43; // [rsp+10h] [rbp-110h]
  __int64 v44; // [rsp+18h] [rbp-108h]
  __int64 v45; // [rsp+18h] [rbp-108h]
  __int64 v46; // [rsp+18h] [rbp-108h]
  _QWORD v47[3]; // [rsp+20h] [rbp-100h] BYREF
  _BYTE v48[8]; // [rsp+38h] [rbp-E8h] BYREF
  __int64 *v49[2]; // [rsp+40h] [rbp-E0h] BYREF
  char v50[16]; // [rsp+50h] [rbp-D0h] BYREF
  __int16 v51; // [rsp+60h] [rbp-C0h]
  __int64 *v52; // [rsp+70h] [rbp-B0h] BYREF
  _BYTE *v53; // [rsp+78h] [rbp-A8h]
  _BYTE *v54; // [rsp+80h] [rbp-A0h]
  __m128i v55; // [rsp+90h] [rbp-90h] BYREF
  __int64 v56; // [rsp+A0h] [rbp-80h] BYREF
  _QWORD *v57; // [rsp+A8h] [rbp-78h]
  __int64 *v58; // [rsp+B0h] [rbp-70h]
  __int64 *v59; // [rsp+B8h] [rbp-68h]
  __int64 v60; // [rsp+C0h] [rbp-60h]
  __int128 v61; // [rsp+C8h] [rbp-58h]
  __int128 v62; // [rsp+D8h] [rbp-48h]
  __int64 v63; // [rsp+E8h] [rbp-38h]

  v8 = a1;
  v9 = a2[3];
  v10 = a2[5];
  v47[0] = a3;
  v44 = v9;
  v41 = v10;
  LOWORD(v56) = 261;
  v47[1] = a4;
  v55.m128i_i64[0] = (__int64)v47;
  v11 = sub_1648B60(120);
  v12 = v11;
  if ( v11 )
    sub_15E2490(v11, a6, a5, (__int64)&v55, v41);
  sub_15E4330(v12, (__int64)a2);
  sub_1560E30((__int64)&v55, **(_QWORD **)(a6 + 16));
  sub_15E0EF0(v12, 0, &v55);
  sub_1814D10(v57);
  v13 = *(_QWORD *)(a1 + 168);
  v55.m128i_i64[0] = (__int64)"entry";
  LOWORD(v56) = 259;
  v14 = (_QWORD *)sub_22077B0(64);
  v15 = (__int64)v14;
  if ( v14 )
    sub_157FB60(v14, v13, (__int64)&v55, v12, 0);
  if ( *(_DWORD *)(a2[3] + 8LL) >> 8 )
  {
    v61 = 0;
    v62 = 0;
    v63 = 0;
    v55.m128i_i64[0] = 0;
    LODWORD(v56) = 0;
    v57 = 0;
    v58 = &v56;
    v59 = &v56;
    v60 = 0;
    v26 = sub_1562A10(&v55, "split-stack", 11, 0, 0);
    sub_15E0EF0(v12, -1, v26);
    sub_1814D10(v57);
    LOWORD(v54) = 257;
    v27 = (_QWORD *)sub_157E9C0(v15);
    v55.m128i_i64[0] = 0;
    v57 = v27;
    v51 = 257;
    v58 = 0;
    LODWORD(v59) = 0;
    v60 = 0;
    *(_QWORD *)&v61 = 0;
    v55.m128i_i64[1] = v15;
    v56 = v15 + 40;
    v28 = (char *)sub_1649960((__int64)a2);
    v30 = sub_15E70A0((__int64)&v55, v28, v29, (__int64)v50, 0);
    v31 = sub_1643350(v57);
    v49[0] = (__int64 *)sub_159C470(v31, 0, 0);
    v49[1] = v49[0];
    v32 = *(_QWORD *)(v30 + 24);
    v48[4] = 0;
    v33 = (__int64 *)sub_15A2E80(v32, v30, v49, 2u, 1u, (__int64)v48, 0);
    v34 = *(_QWORD *)(v8 + 376);
    v49[0] = v33;
    v46 = v34;
    v35 = sub_1648A60(72, 2u);
    v36 = (__int64)v35;
    if ( v35 )
    {
      sub_15F1F50(
        (__int64)v35,
        **(_QWORD **)(*(_QWORD *)(*(_QWORD *)v46 + 24LL) + 16LL),
        54,
        (__int64)(v35 - 6),
        2,
        v15);
      *(_QWORD *)(v36 + 56) = 0;
      sub_15F5B40(v36, *(_QWORD *)(*(_QWORD *)v46 + 24LL), v46, (__int64 *)v49, 1, (__int64)&v52, 0, 0);
    }
    if ( v55.m128i_i64[0] )
      sub_161E7C0((__int64)&v55, v55.m128i_i64[0]);
    v37 = sub_1648A60(56, 0);
    if ( v37 )
      sub_15F82E0((__int64)v37, *(_QWORD *)(v8 + 168), v15);
  }
  else
  {
    v16 = 0;
    v52 = 0;
    v53 = 0;
    v54 = 0;
    v17 = *(_DWORD *)(v44 + 12);
    if ( (*(_BYTE *)(v12 + 18) & 1) != 0 )
    {
      v43 = *(_DWORD *)(v44 + 12);
      sub_15E08E0(v12, 0);
      v16 = v53;
      v17 = v43;
    }
    if ( v17 != 1 )
    {
      v18 = *(_QWORD *)(v12 + 88);
      v19 = v18 + 40LL * (unsigned int)(v17 - 1);
      do
      {
        while ( 1 )
        {
          v55.m128i_i64[0] = v18;
          if ( v54 != v16 )
            break;
          v18 += 40;
          sub_12879C0((__int64)&v52, v16, &v55);
          v16 = v53;
          if ( v18 == v19 )
            goto LABEL_15;
        }
        if ( v16 )
        {
          *(_QWORD *)v16 = v18;
          v16 = v53;
        }
        v16 += 8;
        v18 += 40;
        v53 = v16;
      }
      while ( v18 != v19 );
LABEL_15:
      v8 = a1;
    }
    LOWORD(v56) = 257;
    v42 = v52;
    v38 = (v16 - (_BYTE *)v52) >> 3;
    v20 = sub_1648A60(72, (int)v38 + 1);
    v21 = v20;
    if ( v20 )
    {
      v40 = (__int64)v20;
      sub_15F1F50(
        (__int64)v20,
        **(_QWORD **)(*(_QWORD *)(*a2 + 24LL) + 16LL),
        54,
        (__int64)&v20[-3 * v38 - 3],
        v38 + 1,
        v15);
      *(_QWORD *)(v40 + 56) = 0;
      sub_15F5B40(v40, *(_QWORD *)(*a2 + 24LL), (__int64)a2, v42, v38, (__int64)&v55, 0, 0);
      v21 = (_QWORD *)v40;
    }
    v22 = *(_QWORD *)(v8 + 168);
    if ( *(_BYTE *)(**(_QWORD **)(v44 + 16) + 8LL) )
    {
      v45 = (__int64)v21;
      v23 = sub_1648A60(56, v21 != 0);
      if ( v23 )
        sub_15F7090((__int64)v23, v22, v45, v15);
    }
    else
    {
      v25 = sub_1648A60(56, 0);
      if ( v25 )
        sub_15F7190((__int64)v25, v22, v15);
    }
    if ( v52 )
      j_j___libc_free_0(v52, v54 - (_BYTE *)v52);
  }
  return v12;
}
