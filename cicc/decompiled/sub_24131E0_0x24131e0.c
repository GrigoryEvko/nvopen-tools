// Function: sub_24131E0
// Address: 0x24131e0
//
void __fastcall sub_24131E0(__int64 **a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r15
  __int64 **v3; // rbx
  __int64 v4; // r13
  unsigned __int8 *v5; // rsi
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r13
  unsigned __int8 *v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  int v13; // edx
  unsigned __int8 *v14; // r14
  __int64 v15; // rcx
  __int64 (__fastcall *v16)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  unsigned __int8 *v17; // r11
  __int64 v18; // rax
  __int64 v19; // r13
  int v20; // edx
  unsigned __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r13
  __int64 v25; // rax
  char v26; // r8
  __int64 *v27; // rax
  __int64 *v28; // rax
  __int64 v29; // rax
  __int64 *v30; // rax
  __int64 v31; // rax
  char v32; // r8
  __int64 *v33; // rax
  __int64 *v34; // rax
  __int64 v35; // rax
  __int64 *v36; // rax
  int v37; // edx
  int v38; // edx
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // r14
  unsigned int *v42; // r14
  unsigned int *v43; // rbx
  __int64 v44; // rdx
  unsigned int v45; // esi
  __int64 v46; // rax
  unsigned __int8 *v47; // [rsp+8h] [rbp-168h]
  unsigned __int8 *v48; // [rsp+8h] [rbp-168h]
  __int64 v49; // [rsp+10h] [rbp-160h]
  char v50; // [rsp+10h] [rbp-160h]
  __int64 v51; // [rsp+10h] [rbp-160h]
  char v52; // [rsp+10h] [rbp-160h]
  __int64 v53; // [rsp+10h] [rbp-160h]
  __int64 v54; // [rsp+18h] [rbp-158h]
  __int64 v55; // [rsp+30h] [rbp-140h] BYREF
  __int64 v56; // [rsp+38h] [rbp-138h]
  __int64 v57; // [rsp+40h] [rbp-130h]
  _QWORD v58[4]; // [rsp+50h] [rbp-120h] BYREF
  __int16 v59; // [rsp+70h] [rbp-100h]
  _DWORD v60[8]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v61; // [rsp+A0h] [rbp-D0h]
  unsigned int *v62; // [rsp+B0h] [rbp-C0h] BYREF
  unsigned int v63; // [rsp+B8h] [rbp-B8h]
  char v64; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v65; // [rsp+E8h] [rbp-88h]
  __int64 v66; // [rsp+F0h] [rbp-80h]
  __int64 v67; // [rsp+100h] [rbp-70h]
  __int64 v68; // [rsp+108h] [rbp-68h]
  void *v69; // [rsp+130h] [rbp-40h]

  v2 = a2;
  v3 = a1;
  sub_23D0AB0((__int64)&v62, a2, 0, 0, 0);
  if ( (unsigned __int8)sub_240D530() )
  {
    v38 = *(_DWORD *)(a2 + 4);
    v61 = 257;
    v39 = v38 & 0x7FFFFFF;
    v59 = 257;
    v55 = *(_QWORD *)(a2 - 32 * v39);
    v56 = *(_QWORD *)(a2 + 32 * (1 - v39));
    v57 = sub_921630(&v62, *(_QWORD *)(a2 + 32 * (2 - v39)), *(_QWORD *)(**a1 + 64), 0, (__int64)v58);
    v40 = **a1;
    a2 = *(_QWORD *)(v40 + 552);
    sub_921880(&v62, a2, *(_QWORD *)(v40 + 560), (int)&v55, 3, (__int64)v60, 0);
  }
  v4 = **a1;
  v5 = sub_BD3990(*(unsigned __int8 **)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF)), a2);
  v54 = sub_2412E00(v4, (unsigned __int64)v5, v2 + 24, 0, v6, v7);
  v8 = **a1;
  v9 = sub_BD3990(*(unsigned __int8 **)(v2 + 32 * (1LL - (*(_DWORD *)(v2 + 4) & 0x7FFFFFF))), (__int64)v5);
  v12 = sub_2412E00(v8, (unsigned __int64)v9, v2 + 24, 0, v10, v11);
  v13 = *(_DWORD *)(v2 + 4);
  v49 = v12;
  v59 = 257;
  v14 = (unsigned __int8 *)sub_AD64C0(*(_QWORD *)(*(_QWORD *)(v2 + 32 * (2LL - (v13 & 0x7FFFFFF))) + 8LL), 1, 0);
  v15 = 32 * (2LL - (*(_DWORD *)(v2 + 4) & 0x7FFFFFF));
  v16 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v67 + 32LL);
  v17 = *(unsigned __int8 **)(v2 + v15);
  if ( v16 != sub_9201A0 )
  {
    v48 = *(unsigned __int8 **)(v2 + v15);
    v46 = v16(v67, 17u, v17, v14, 0, 0);
    v17 = v48;
    v19 = v46;
    goto LABEL_9;
  }
  if ( *v17 <= 0x15u && *v14 <= 0x15u )
  {
    v47 = *(unsigned __int8 **)(v2 + v15);
    if ( (unsigned __int8)sub_AC47B0(17) )
      v18 = sub_AD5570(17, (__int64)v47, v14, 0, 0);
    else
      v18 = sub_AABE40(0x11u, v47, v14);
    v17 = v47;
    v19 = v18;
LABEL_9:
    if ( v19 )
      goto LABEL_10;
  }
  v61 = 257;
  v19 = sub_B504D0(17, (__int64)v17, (__int64)v14, (__int64)v60, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v68 + 16LL))(
    v68,
    v19,
    v58,
    v65,
    v66);
  v41 = 4LL * v63;
  if ( v62 != &v62[v41] )
  {
    v42 = &v62[v41];
    v43 = v62;
    do
    {
      v44 = *((_QWORD *)v43 + 1);
      v45 = *v43;
      v43 += 4;
      sub_B99FD0(v19, v45, v44);
    }
    while ( v42 != v43 );
    v3 = a1;
  }
LABEL_10:
  v20 = *(_DWORD *)(v2 + 4);
  v21 = *(_QWORD *)(v2 + 80);
  v61 = 257;
  v58[0] = v54;
  v58[2] = v19;
  v58[1] = v49;
  v22 = 3LL - (v20 & 0x7FFFFFF);
  v23 = *(_QWORD *)(v2 - 32);
  v58[3] = *(_QWORD *)(v2 + 32 * v22);
  v24 = sub_921880(&v62, v21, v23, (int)v58, 4, (__int64)v60, 0);
  LOWORD(v25) = sub_A74840((_QWORD *)(v2 + 72), 0);
  v26 = 0;
  if ( BYTE1(v25) )
  {
    if ( (_BYTE)qword_4FE3A68 )
    {
      v26 = -1;
      if ( 1LL << v25 )
      {
        _BitScanReverse64((unsigned __int64 *)&v25, 1LL << v25);
        v26 = 63 - (v25 ^ 0x3F);
      }
    }
  }
  v50 = v26;
  v27 = (__int64 *)sub_BD5C60(v24);
  *(_QWORD *)(v24 + 72) = sub_A7B980((__int64 *)(v24 + 72), v27, 1, 86);
  v28 = (__int64 *)sub_BD5C60(v24);
  v29 = sub_A77A40(v28, v50);
  v60[0] = 0;
  v51 = v29;
  v30 = (__int64 *)sub_BD5C60(v24);
  *(_QWORD *)(v24 + 72) = sub_A7B660((__int64 *)(v24 + 72), v30, v60, 1, v51);
  LOWORD(v31) = sub_A74840((_QWORD *)(v2 + 72), 1);
  v32 = 0;
  if ( BYTE1(v31) )
  {
    if ( (_BYTE)qword_4FE3A68 )
    {
      v32 = -1;
      if ( 1LL << v31 )
      {
        _BitScanReverse64((unsigned __int64 *)&v31, 1LL << v31);
        v32 = 63 - (v31 ^ 0x3F);
      }
    }
  }
  v52 = v32;
  v33 = (__int64 *)sub_BD5C60(v24);
  *(_QWORD *)(v24 + 72) = sub_A7B980((__int64 *)(v24 + 72), v33, 2, 86);
  v34 = (__int64 *)sub_BD5C60(v24);
  v35 = sub_A77A40(v34, v52);
  v60[0] = 1;
  v53 = v35;
  v36 = (__int64 *)sub_BD5C60(v24);
  *(_QWORD *)(v24 + 72) = sub_A7B660((__int64 *)(v24 + 72), v36, v60, 1, v53);
  if ( (_BYTE)qword_4FE3408 )
  {
    v61 = 257;
    v59 = 257;
    v37 = *(_DWORD *)(v2 + 4);
    v55 = v54;
    v56 = sub_A830B0(&v62, *(_QWORD *)(v2 + 32 * (2LL - (v37 & 0x7FFFFFF))), *(_QWORD *)(**v3 + 64), (__int64)v58);
    sub_921880(&v62, *(_QWORD *)(**v3 + 424), *(_QWORD *)(**v3 + 432), (int)&v55, 2, (__int64)v60, 0);
  }
  nullsub_61();
  v69 = &unk_49DA100;
  nullsub_63();
  if ( v62 != (unsigned int *)&v64 )
    _libc_free((unsigned __int64)v62);
}
