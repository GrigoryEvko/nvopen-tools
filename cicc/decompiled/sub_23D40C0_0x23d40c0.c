// Function: sub_23D40C0
// Address: 0x23d40c0
//
__int64 __fastcall sub_23D40C0(__int64 a1)
{
  __int64 v1; // r13
  unsigned int v2; // r15d
  __int64 v4; // r14
  __int64 v5; // rax
  unsigned __int64 v6; // rbx
  unsigned int v7; // eax
  _BYTE *v8; // rsi
  unsigned int v9; // edx
  unsigned int v10; // eax
  unsigned int v11; // eax
  unsigned __int64 v12; // r14
  unsigned int v13; // eax
  unsigned int v14; // r9d
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r11
  __int64 v22; // r10
  _BYTE *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rsi
  unsigned int v27; // [rsp+8h] [rbp-168h]
  unsigned int v28; // [rsp+10h] [rbp-160h]
  unsigned int v29; // [rsp+18h] [rbp-158h]
  unsigned __int8 v30; // [rsp+1Fh] [rbp-151h]
  __int64 v31; // [rsp+20h] [rbp-150h]
  __int64 v32; // [rsp+20h] [rbp-150h]
  unsigned int v33; // [rsp+28h] [rbp-148h]
  __int64 v34; // [rsp+28h] [rbp-148h]
  _BYTE *v35; // [rsp+30h] [rbp-140h]
  __int64 v36; // [rsp+30h] [rbp-140h]
  __int64 v37; // [rsp+30h] [rbp-140h]
  __int64 v38; // [rsp+38h] [rbp-138h]
  __int64 v39; // [rsp+48h] [rbp-128h] BYREF
  __int64 v40; // [rsp+50h] [rbp-120h] BYREF
  __int64 v41; // [rsp+58h] [rbp-118h] BYREF
  __int64 v42; // [rsp+60h] [rbp-110h] BYREF
  unsigned int v43; // [rsp+68h] [rbp-108h]
  int v44; // [rsp+6Ch] [rbp-104h]
  _QWORD v45[2]; // [rsp+70h] [rbp-100h] BYREF
  _BYTE v46[32]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v47; // [rsp+A0h] [rbp-D0h]
  unsigned __int64 v48; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 *v49; // [rsp+B8h] [rbp-B8h]
  __int64 *v50; // [rsp+C0h] [rbp-B0h]
  __int64 *v51; // [rsp+C8h] [rbp-A8h]
  __int64 *v52; // [rsp+D0h] [rbp-A0h]
  __int64 v53; // [rsp+D8h] [rbp-98h]
  __int64 *v54; // [rsp+E0h] [rbp-90h]
  __int64 *v55; // [rsp+E8h] [rbp-88h]
  __int64 *v56; // [rsp+F0h] [rbp-80h]
  __int64 *v57; // [rsp+F8h] [rbp-78h]

  if ( *(_BYTE *)a1 != 61 )
    return 0;
  v1 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(v1 + 8) != 12 )
    return 0;
  v4 = *(_QWORD *)(a1 - 32);
  if ( *(_BYTE *)v4 != 63 )
    return 0;
  if ( !sub_B4DE30(*(_QWORD *)(a1 - 32)) )
    return 0;
  if ( (*(_DWORD *)(v4 + 4) & 0x7FFFFFF) != 3 )
    return 0;
  v5 = *(_QWORD *)(v4 + 72);
  if ( *(_BYTE *)(v5 + 8) != 16 )
    return 0;
  v6 = (*(_QWORD *)(v5 + 32) - 32LL) & 0xFFFFFFFFFFFFFFDFLL;
  if ( v6 )
    return 0;
  if ( **(_BYTE **)(v4 - 96) != 3 )
    return 0;
  v38 = *(_QWORD *)(v4 - 96);
  LOBYTE(v7) = sub_B2FC80(v38);
  v2 = v7;
  if ( (_BYTE)v7 )
    return 0;
  if ( (*(_BYTE *)(v38 + 80) & 1) == 0 )
    return 0;
  v35 = *(_BYTE **)(v38 - 32);
  if ( *v35 != 15 )
    return 0;
  v48 = 0;
  if ( !(unsigned __int8)sub_10081F0((__int64 **)&v48, *(_QWORD *)(v4 + 32 * (1LL - (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)))) )
    return 0;
  v8 = *(_BYTE **)(v4 + 32 * (1LL - (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)) + 32);
  v51 = &v40;
  v48 = 0;
  v49 = &v39;
  v50 = &v39;
  v52 = &v41;
  v53 = 0;
  v54 = &v39;
  v55 = &v39;
  v56 = &v40;
  v57 = &v41;
  v30 = sub_23D3BD0((__int64)&v48, v8);
  if ( !v30 )
    return 0;
  v9 = sub_BCB060(*(_QWORD *)(v39 + 8));
  v33 = (v9 - 32) & 0xFFFFFFDF;
  if ( v33 )
    return 0;
  _BitScanReverse(&v10, v9);
  v11 = v10 ^ 0x1F;
  if ( !v9 )
    v11 = 32;
  if ( v9 + v11 - 31 != v41 && v41 != v9 + v11 - 32 )
    return 0;
  v27 = v41;
  v12 = v9;
  v28 = v9;
  v31 = v40;
  v13 = sub_AC5290((__int64)v35);
  v29 = v13;
  if ( v12 > v13 || v13 > 2 * v12 )
    return v2;
  v14 = v27;
  LODWORD(v49) = v28;
  if ( v28 > 0x40 )
  {
    sub_C43690((__int64)&v48, 0, 0);
    v14 = v27;
  }
  else
  {
    v48 = 0;
  }
  if ( v14 != (_DWORD)v49 )
  {
    if ( v14 > 0x3F || (unsigned int)v49 > 0x40 )
      sub_C43C90(&v48, v14, (unsigned int)v49);
    else
      v48 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v14 - (unsigned __int8)v49 + 64) << v14;
  }
  while ( v29 > (unsigned int)v6 )
  {
    v15 = sub_AC5320((__int64)v35, v6);
    if ( v12 > v15 )
    {
      v16 = v48;
      v17 = v31 << v15;
      if ( (unsigned int)v49 > 0x40 )
        v16 = *(_QWORD *)v48;
      if ( v6 == (v16 & v17) >> v27 )
        ++v33;
    }
    ++v6;
  }
  if ( (unsigned int)v49 > 0x40 && v48 )
    j_j___libc_free_0_0(v48);
  if ( v12 != v33 )
    return 0;
  v18 = sub_AC5320((__int64)v35, 0);
  sub_23D0AB0((__int64)&v48, a1, 0, 0, 0);
  v19 = sub_BCB2A0(v57);
  v20 = sub_ACD640(v19, v12 != v18, 0);
  v44 = 0;
  v21 = *(_QWORD *)(v39 + 8);
  v45[1] = v20;
  v42 = v21;
  v36 = v21;
  v47 = 257;
  v45[0] = v39;
  v22 = sub_B33D10((__int64)&v48, 0x43u, (__int64)&v42, 1, (int)v45, 2, v43, (__int64)v46);
  if ( v12 == v18 )
  {
    v47 = 257;
    v26 = sub_A830B0((unsigned int **)&v48, v22, v1, (__int64)v46);
  }
  else
  {
    v32 = v22;
    v34 = v36;
    v47 = 257;
    v23 = (_BYTE *)sub_AD64C0(v36, 0, 0);
    v37 = sub_92B530((unsigned int **)&v48, 0x20u, v39, v23, (__int64)v46);
    v47 = 257;
    v24 = sub_AD64C0(v34, v18, 0);
    v25 = sub_B36550((unsigned int **)&v48, v37, v24, v32, (__int64)v46, 0);
    v47 = 257;
    v26 = sub_A830B0((unsigned int **)&v48, v25, v1, (__int64)v46);
  }
  sub_BD84D0(a1, v26);
  sub_F94A20(&v48, v26);
  return v30;
}
