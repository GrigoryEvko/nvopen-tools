// Function: sub_33A10A0
// Address: 0x33a10a0
//
void __fastcall sub_33A10A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v8; // rdx
  __int64 v9; // rax
  unsigned __int16 v10; // r13
  unsigned __int8 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // r13d
  __int64 v15; // r14
  __int64 v16; // rax
  unsigned int v17; // eax
  bool v18; // zf
  __int64 v19; // rdx
  unsigned __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned __int64 v23; // rax
  unsigned __int8 v24; // r8
  unsigned __int16 v25; // r14
  __int16 v26; // cx
  _QWORD *v27; // r10
  int v28; // r15d
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdi
  __int64 v32; // rdx
  char v33; // al
  unsigned __int64 v34; // rcx
  int v35; // edx
  __int64 v36; // rax
  __int64 v37; // rcx
  int v38; // eax
  int v39; // r13d
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r10
  __int64 v44; // r8
  __int64 v45; // r11
  __int64 v46; // rax
  __int64 v47; // rax
  unsigned int v48; // edx
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r13
  int v52; // edx
  _QWORD *v53; // rax
  __int64 v54; // r12
  __int128 v55; // [rsp-20h] [rbp-140h]
  __int64 v56; // [rsp+8h] [rbp-118h]
  _QWORD *v57; // [rsp+10h] [rbp-110h]
  __int64 v58; // [rsp+18h] [rbp-108h]
  __int128 v59; // [rsp+20h] [rbp-100h]
  __int64 v60; // [rsp+30h] [rbp-F0h]
  unsigned __int64 v61; // [rsp+30h] [rbp-F0h]
  unsigned __int8 v62; // [rsp+30h] [rbp-F0h]
  __int64 v63; // [rsp+30h] [rbp-F0h]
  int v64; // [rsp+30h] [rbp-F0h]
  __int64 v65; // [rsp+38h] [rbp-E8h]
  __int64 v66; // [rsp+70h] [rbp-B0h] BYREF
  int v67; // [rsp+78h] [rbp-A8h]
  unsigned int v68; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v69; // [rsp+88h] [rbp-98h]
  __int64 v70; // [rsp+90h] [rbp-90h]
  __int64 v71; // [rsp+98h] [rbp-88h]
  __int64 v72; // [rsp+A0h] [rbp-80h]
  __int64 v73; // [rsp+A8h] [rbp-78h]
  __int128 v74; // [rsp+B0h] [rbp-70h]
  __int64 v75; // [rsp+C0h] [rbp-60h]
  __int64 v76; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v77; // [rsp+D8h] [rbp-48h]
  __int64 v78; // [rsp+E0h] [rbp-40h]
  __int64 v79; // [rsp+E8h] [rbp-38h]

  v6 = a2;
  v8 = *(unsigned int *)(a1 + 848);
  v9 = *(_QWORD *)a1;
  v66 = 0;
  v67 = v8;
  if ( v9 )
  {
    v8 = v9 + 48;
    if ( &v66 != (__int64 *)(v9 + 48) )
    {
      a2 = *(_QWORD *)(v9 + 48);
      v66 = a2;
      if ( a2 )
        sub_B96E90((__int64)&v66, a2, 1);
    }
  }
  v10 = *(_WORD *)(v6 + 2);
  v11 = *(_BYTE *)(v6 + 72);
  *(_QWORD *)&v59 = sub_33738B0(a1, a2, v8, a4, a5, a6);
  v12 = *(_QWORD *)(a1 + 864);
  *((_QWORD *)&v59 + 1) = v13;
  v14 = (v10 >> 7) & 7;
  v15 = *(_QWORD *)(v12 + 16);
  v60 = *(_QWORD *)(*(_QWORD *)(v6 - 64) + 8LL);
  v16 = sub_2E79000(*(__int64 **)(v12 + 40));
  v17 = sub_336EEB0(v15, v16, v60, 0);
  v18 = *(_BYTE *)(v15 + 100) == 0;
  v68 = v17;
  v69 = v19;
  if ( v18 )
  {
    _BitScanReverse64(&v20, 1LL << (*(_WORD *)(v6 + 2) >> 1));
    v61 = 0x8000000000000000LL >> ((unsigned __int8)v20 ^ 0x3Fu);
    if ( (_WORD)v68 )
    {
      if ( (_WORD)v68 == 1 || (unsigned __int16)(v68 - 504) <= 7u )
        goto LABEL_36;
      v22 = 16LL * ((unsigned __int16)v68 - 1);
      v21 = *(_QWORD *)&byte_444C4A0[v22];
      LOBYTE(v22) = byte_444C4A0[v22 + 8];
    }
    else
    {
      v21 = sub_3007260((__int64)&v68);
      v70 = v21;
      v71 = v22;
    }
    v76 = v21;
    LOBYTE(v77) = v22;
    if ( (unsigned __int64)sub_CA1930(&v76) >> 3 > v61 )
      sub_C64ED0("Cannot generate unaligned atomic store", 1u);
  }
  sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL));
  LOWORD(v23) = sub_2FEC5A0(v15, v6);
  v24 = v11;
  v25 = v23;
  v26 = *(_WORD *)(v6 + 2) >> 1;
  v27 = *(_QWORD **)(*(_QWORD *)(a1 + 864) + 40LL);
  v76 = 0;
  v77 = 0;
  _BitScanReverse64(&v23, 1LL << v26);
  v78 = 0;
  v79 = 0;
  v28 = 63 - (v23 ^ 0x3F);
  if ( !(_WORD)v68 )
  {
    v57 = v27;
    v62 = v24;
    v29 = sub_3007260((__int64)&v68);
    v24 = v62;
    v27 = v57;
    v31 = v30;
    v72 = v29;
    v32 = v29;
    v73 = v31;
    v33 = v31;
    goto LABEL_11;
  }
  if ( (_WORD)v68 == 1 || (unsigned __int16)(v68 - 504) <= 7u )
LABEL_36:
    BUG();
  v32 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v68 - 16];
  v33 = byte_444C4A0[16 * (unsigned __int16)v68 - 8];
LABEL_11:
  v34 = (unsigned __int64)(v32 + 7) >> 3;
  v35 = v34;
  v18 = v33 == 0;
  v36 = *(_QWORD *)(v6 - 32);
  if ( v18 )
    v35 = v34;
  if ( v36 )
  {
    *((_QWORD *)&v74 + 1) = 0;
    BYTE4(v75) = 0;
    *(_QWORD *)&v74 = v36 & 0xFFFFFFFFFFFFFFFBLL;
    v37 = *(_QWORD *)(v36 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v37 + 8) - 17 <= 1 )
      v37 = **(_QWORD **)(v37 + 16);
    v38 = *(_DWORD *)(v37 + 8) >> 8;
  }
  else
  {
    v74 = 0u;
    v38 = 0;
    BYTE4(v75) = 0;
  }
  LODWORD(v75) = v38;
  v39 = sub_2E7BD70(v27, v25, v35, v28, (int)&v76, 0, v74, v75, v24, v14, 0);
  v40 = sub_338B750(a1, *(_QWORD *)(v6 - 64));
  v42 = (unsigned int)v41;
  v43 = v40;
  v44 = v40;
  v45 = v41;
  v46 = *(_QWORD *)(v40 + 48) + 16LL * (unsigned int)v41;
  if ( *(_WORD *)v46 != (_WORD)v68 || v69 != *(_QWORD *)(v46 + 8) && !*(_WORD *)v46 )
  {
    v65 = v41;
    v47 = sub_33FB4C0(*(_QWORD *)(a1 + 864), v43, v41, &v66, v68, v69);
    v45 = v65;
    v44 = v47;
    v42 = v48;
  }
  v56 = v42;
  v58 = v45;
  v63 = v44;
  v49 = sub_338B750(a1, *(_QWORD *)(v6 - 32));
  *((_QWORD *)&v55 + 1) = v56 | v58 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v55 = v63;
  v51 = sub_33F34C0(*(_QWORD *)(a1 + 864), 339, (unsigned int)&v66, v68, v69, v39, v59, v55, v49, v50);
  v64 = v52;
  v76 = v6;
  v53 = sub_337DC20(a1 + 8, &v76);
  *v53 = v51;
  *((_DWORD *)v53 + 2) = v64;
  v54 = *(_QWORD *)(a1 + 864);
  if ( v51 )
  {
    nullsub_1875(v51, *(_QWORD *)(a1 + 864), 0);
    *(_QWORD *)(v54 + 384) = v51;
    *(_DWORD *)(v54 + 392) = v64;
    sub_33E2B60(v54, 0);
  }
  else
  {
    *(_QWORD *)(v54 + 384) = 0;
    *(_DWORD *)(v54 + 392) = v64;
  }
  if ( v66 )
    sub_B91220((__int64)&v66, v66);
}
