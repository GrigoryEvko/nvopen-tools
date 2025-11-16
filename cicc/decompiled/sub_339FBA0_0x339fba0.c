// Function: sub_339FBA0
// Address: 0x339fba0
//
void __fastcall sub_339FBA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // rax
  unsigned __int8 v10; // r14
  __int64 v11; // rax
  __int64 *v12; // r15
  __int64 v13; // rax
  __int64 v14; // rdx
  _BYTE *v15; // r12
  __int64 v16; // rax
  unsigned int v17; // eax
  __int64 v18; // r15
  __int64 v19; // rdx
  __int64 v20; // rax
  int v21; // eax
  bool v22; // zf
  __int64 v23; // rdx
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // r15
  __int64 v26; // rdx
  __int64 v27; // rdx
  char v28; // al
  __int64 v29; // r15
  __int64 v30; // rax
  unsigned __int16 v31; // r15
  __int16 v32; // cx
  _QWORD *v33; // r11
  __int64 v34; // rax
  int v35; // ecx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdi
  __int64 v39; // rdx
  char v40; // al
  unsigned __int64 v41; // rsi
  int v42; // edx
  __int64 v43; // rax
  __int64 v44; // rdi
  int v45; // eax
  int v46; // r14d
  __int64 (__fastcall *v47)(__int64, __int64); // rax
  __int64 v48; // rsi
  unsigned int v49; // ecx
  __int128 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // r12
  int v53; // r14d
  __int64 v54; // rcx
  int v55; // edx
  _QWORD *v56; // rax
  __int64 v57; // r14
  __int64 v58; // rax
  __int64 v59; // rax
  unsigned int v60; // edx
  __int128 v61; // [rsp-20h] [rbp-150h]
  __int64 v62; // [rsp+0h] [rbp-130h]
  __int64 v63; // [rsp+8h] [rbp-128h]
  _QWORD *v64; // [rsp+8h] [rbp-128h]
  int v65; // [rsp+10h] [rbp-120h]
  int v66; // [rsp+14h] [rbp-11Ch]
  unsigned int v67; // [rsp+18h] [rbp-118h]
  __int64 v68; // [rsp+20h] [rbp-110h]
  __int64 v69; // [rsp+20h] [rbp-110h]
  __int64 v70; // [rsp+28h] [rbp-108h]
  unsigned __int64 v71; // [rsp+28h] [rbp-108h]
  __int64 v72; // [rsp+80h] [rbp-B0h] BYREF
  int v73; // [rsp+88h] [rbp-A8h]
  __int64 v74; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v75; // [rsp+98h] [rbp-98h]
  __int64 v76; // [rsp+A0h] [rbp-90h]
  __int64 v77; // [rsp+A8h] [rbp-88h]
  __int64 v78; // [rsp+B0h] [rbp-80h]
  __int64 v79; // [rsp+B8h] [rbp-78h]
  __int128 v80; // [rsp+C0h] [rbp-70h]
  __int64 v81; // [rsp+D0h] [rbp-60h]
  __int64 v82; // [rsp+E0h] [rbp-50h] BYREF
  __int64 v83; // [rsp+E8h] [rbp-48h]
  __int64 v84; // [rsp+F0h] [rbp-40h]
  __int64 v85; // [rsp+F8h] [rbp-38h]

  v7 = a2;
  v8 = *(unsigned int *)(a1 + 848);
  v9 = *(_QWORD *)a1;
  v72 = 0;
  v73 = v8;
  if ( v9 )
  {
    v8 = v9 + 48;
    if ( &v72 != (__int64 *)(v9 + 48) )
    {
      a2 = *(_QWORD *)(v9 + 48);
      v72 = a2;
      if ( a2 )
        sub_B96E90((__int64)&v72, a2, 1);
    }
  }
  v10 = *(_BYTE *)(v7 + 72);
  v66 = (*(_WORD *)(v7 + 2) >> 7) & 7;
  v11 = sub_33738B0(a1, a2, v8, a4, a5, a6);
  v12 = *(__int64 **)(v7 + 8);
  v68 = v11;
  v13 = *(_QWORD *)(a1 + 864);
  v70 = v14;
  v15 = *(_BYTE **)(v13 + 16);
  v16 = sub_2E79000(*(__int64 **)(v13 + 40));
  v17 = sub_2D5BAE0((__int64)v15, v16, v12, 0);
  v18 = *(_QWORD *)(v7 + 8);
  v67 = v17;
  v62 = v19;
  v20 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL));
  v21 = sub_336EEB0((__int64)v15, v20, v18, 0);
  v22 = v15[100] == 0;
  LODWORD(v74) = v21;
  v75 = v23;
  if ( v22 )
  {
    _BitScanReverse64(&v24, 1LL << (*(_WORD *)(v7 + 2) >> 1));
    v25 = 0x8000000000000000LL >> ((unsigned __int8)v24 ^ 0x3Fu);
    if ( (_WORD)v74 )
    {
      if ( (_WORD)v74 == 1 || (unsigned __int16)(v74 - 504) <= 7u )
        goto LABEL_39;
      v58 = 16LL * ((unsigned __int16)v74 - 1);
      v27 = *(_QWORD *)&byte_444C4A0[v58];
      v28 = byte_444C4A0[v58 + 8];
    }
    else
    {
      v76 = sub_3007260((__int64)&v74);
      v77 = v26;
      v27 = v76;
      v28 = v77;
    }
    v82 = v27;
    LOBYTE(v83) = v28;
    if ( (unsigned __int64)sub_CA1930(&v82) >> 3 > v25 )
      sub_C64ED0("Cannot generate unaligned atomic load", 1u);
  }
  v29 = *(_QWORD *)(a1 + 880);
  v63 = *(_QWORD *)(a1 + 888);
  v30 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL));
  v31 = sub_2FEC4A0((__int64)v15, v7, v30, v29, v63);
  v32 = *(_WORD *)(v7 + 2) >> 1;
  v33 = *(_QWORD **)(*(_QWORD *)(a1 + 864) + 40LL);
  v82 = 0;
  v83 = 0;
  _BitScanReverse64((unsigned __int64 *)&v34, 1LL << v32);
  v84 = 0;
  v85 = 0;
  v35 = 63 - (v34 ^ 0x3F);
  if ( !(_WORD)v74 )
  {
    v65 = 63 - (v34 ^ 0x3F);
    v64 = v33;
    v36 = sub_3007260((__int64)&v74);
    v33 = v64;
    v35 = v65;
    v38 = v37;
    v78 = v36;
    v39 = v36;
    v79 = v38;
    v40 = v38;
    goto LABEL_11;
  }
  if ( (_WORD)v74 == 1 || (unsigned __int16)(v74 - 504) <= 7u )
LABEL_39:
    BUG();
  v39 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v74 - 16];
  v40 = byte_444C4A0[16 * (unsigned __int16)v74 - 8];
LABEL_11:
  v41 = (unsigned __int64)(v39 + 7) >> 3;
  v42 = v41;
  v22 = v40 == 0;
  v43 = *(_QWORD *)(v7 - 32);
  if ( v22 )
    v42 = v41;
  if ( v43 )
  {
    *((_QWORD *)&v80 + 1) = 0;
    BYTE4(v81) = 0;
    *(_QWORD *)&v80 = v43 & 0xFFFFFFFFFFFFFFFBLL;
    v44 = *(_QWORD *)(v43 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v44 + 8) - 17 <= 1 )
      v44 = **(_QWORD **)(v44 + 16);
    v45 = *(_DWORD *)(v44 + 8) >> 8;
  }
  else
  {
    v80 = 0u;
    v45 = 0;
    BYTE4(v81) = 0;
  }
  LODWORD(v81) = v45;
  v46 = sub_2E7BD70(v33, v31, v42, v35, (int)&v82, 0, v80, v81, v10, v66, 0);
  v47 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v15 + 2400LL);
  if ( v47 == sub_302E280 )
  {
    v48 = v68;
    v49 = v70;
  }
  else
  {
    v59 = ((__int64 (__fastcall *)(_BYTE *, __int64, __int64, __int64 *, _QWORD))v47)(
            v15,
            v68,
            v70,
            &v72,
            *(_QWORD *)(a1 + 864));
    v49 = v60;
    v48 = v59;
  }
  v71 = v49 | v70 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v50 = sub_338B750(a1, *(_QWORD *)(v7 - 32));
  *((_QWORD *)&v61 + 1) = v71;
  *(_QWORD *)&v61 = v48;
  v52 = sub_33E6F50(*(_QWORD *)(a1 + 864), 338, (unsigned int)&v72, v74, v75, v46, v74, v75, v61, v50);
  v53 = v51;
  if ( (_WORD)v74 != (_WORD)v67 || (v54 = v52, !(_WORD)v74) && v75 != v62 )
  {
    v54 = sub_33FB4C0(*(_QWORD *)(a1 + 864), v52, v51, &v72, v67, v62);
    v53 = v55;
  }
  v69 = v54;
  v82 = v7;
  v56 = sub_337DC20(a1 + 8, &v82);
  *v56 = v69;
  *((_DWORD *)v56 + 2) = v53;
  v57 = *(_QWORD *)(a1 + 864);
  if ( v52 )
  {
    nullsub_1875(v52, *(_QWORD *)(a1 + 864), 0);
    *(_QWORD *)(v57 + 384) = v52;
    *(_DWORD *)(v57 + 392) = 1;
    sub_33E2B60(v57, 0);
  }
  else
  {
    *(_QWORD *)(v57 + 384) = 0;
    *(_DWORD *)(v57 + 392) = 1;
  }
  if ( v72 )
    sub_B91220((__int64)&v72, v72);
}
