// Function: sub_339CDA0
// Address: 0x339cda0
//
void __fastcall sub_339CDA0(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  __int64 v5; // rax
  int v6; // edx
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // r13
  int v11; // r14d
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rdx
  unsigned __int16 *v17; // rax
  __int64 v18; // r8
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int16 v22; // r10
  int v23; // eax
  __int64 v24; // rax
  _QWORD *v25; // r15
  __int64 v26; // rax
  __int64 v27; // r15
  int v28; // eax
  int v29; // edx
  __int64 v30; // rax
  __int64 v31; // r8
  __int64 v32; // r15
  int v33; // edx
  _QWORD *v34; // rax
  unsigned __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rax
  int v39; // edx
  __int128 v40; // [rsp+0h] [rbp-130h]
  __int128 v41; // [rsp+10h] [rbp-120h]
  unsigned __int64 v42; // [rsp+28h] [rbp-108h]
  unsigned __int16 v43; // [rsp+30h] [rbp-100h]
  __int64 v44; // [rsp+30h] [rbp-100h]
  _QWORD *v45; // [rsp+38h] [rbp-F8h]
  void (__fastcall *v46)(__int128 *, __int64, __int64); // [rsp+38h] [rbp-F8h]
  __int64 v47; // [rsp+40h] [rbp-F0h]
  __int64 v48; // [rsp+48h] [rbp-E8h]
  __int64 v49; // [rsp+50h] [rbp-E0h]
  __int64 v50; // [rsp+50h] [rbp-E0h]
  __int64 v51; // [rsp+58h] [rbp-D8h]
  __int64 v52; // [rsp+60h] [rbp-D0h]
  __int64 v53; // [rsp+60h] [rbp-D0h]
  __int64 v54; // [rsp+68h] [rbp-C8h]
  int v56; // [rsp+78h] [rbp-B8h]
  __int64 (__fastcall *v57)(_QWORD *, _QWORD, __int64 *, __int64, __int64, unsigned __int64, __int64, __int64, __int64, __int64, _QWORD, _QWORD); // [rsp+78h] [rbp-B8h]
  __int64 v58; // [rsp+B0h] [rbp-80h] BYREF
  int v59; // [rsp+B8h] [rbp-78h]
  __int128 v60; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v61; // [rsp+D0h] [rbp-60h]
  __int64 v62; // [rsp+E0h] [rbp-50h] BYREF
  int v63; // [rsp+E8h] [rbp-48h]

  v5 = *(_QWORD *)a1;
  v6 = *(_DWORD *)(a1 + 848);
  v58 = 0;
  v59 = v6;
  if ( v5 )
  {
    if ( &v58 != (__int64 *)(v5 + 48) )
    {
      v7 = *(_QWORD *)(v5 + 48);
      v58 = v7;
      if ( v7 )
        sub_B96E90((__int64)&v58, v7, 1);
    }
  }
  v8 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v49 = *(_QWORD *)(a2 - 32 * v8);
  v9 = *(_QWORD *)(a2 + 32 * (1 - v8));
  v10 = *(_QWORD *)(a2 + 32 * (2 - v8));
  if ( a3 )
  {
    v11 = 0;
    LOWORD(v12) = sub_A74840((_QWORD *)(a2 + 72), 1);
    if ( BYTE1(v12) )
      v11 = v12;
  }
  else
  {
    v35 = *(_QWORD *)(v10 + 24);
    if ( *(_DWORD *)(v10 + 32) > 0x40u )
      v35 = *(_QWORD *)v35;
    v11 = 0;
    if ( v35 )
    {
      _BitScanReverse64(&v35, v35);
      v11 = 63 - (v35 ^ 0x3F);
    }
    v10 = *(_QWORD *)(a2 + 32 * (3 - v8));
  }
  v52 = sub_338B750(a1, v9);
  v54 = v13;
  v50 = sub_338B750(a1, v49);
  v51 = v14;
  *(_QWORD *)&v41 = sub_338B750(a1, v10);
  v15 = *(_QWORD *)(a1 + 864);
  *((_QWORD *)&v41 + 1) = v16;
  v17 = (unsigned __int16 *)(*(_QWORD *)(v52 + 48) + 16LL * (unsigned int)v54);
  v18 = *((_QWORD *)v17 + 1);
  v19 = *v17;
  v62 = 0;
  v63 = 0;
  *(_QWORD *)&v40 = sub_33F17F0(v15, 51, &v62, v19, v18);
  *((_QWORD *)&v40 + 1) = v20;
  if ( v62 )
    sub_B91220((__int64)&v62, v62);
  v47 = *(unsigned __int16 *)(*(_QWORD *)(v50 + 48) + 16LL * (unsigned int)v51);
  v48 = *(_QWORD *)(*(_QWORD *)(v50 + 48) + 16LL * (unsigned int)v51 + 8);
  if ( (*(_BYTE *)(a2 + 7) & 0x20) == 0 || (v21 = sub_B91C10(a2, 9), v22 = 10, !v21) )
    v22 = 2;
  v43 = v22;
  v45 = *(_QWORD **)(*(_QWORD *)(a1 + 864) + 40LL);
  sub_B91FC0(&v62, a2);
  BYTE4(v61) = 0;
  *((_QWORD *)&v60 + 1) = 0;
  *(_QWORD *)&v60 = v9 & 0xFFFFFFFFFFFFFFFBLL;
  v23 = 0;
  if ( v9 )
  {
    v24 = *(_QWORD *)(v9 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17 <= 1 )
      v24 = **(_QWORD **)(v24 + 16);
    v23 = *(_DWORD *)(v24 + 8) >> 8;
  }
  LODWORD(v61) = v23;
  v42 = sub_2E7BD70(v45, v43, -1, v11, (int)&v62, 0, v60, v61, 1u, 0, 0);
  v25 = *(_QWORD **)(*(_QWORD *)(a1 + 864) + 16LL);
  v44 = v25[1];
  v46 = *(void (__fastcall **)(__int128 *, __int64, __int64))(*(_QWORD *)v44 + 104LL);
  v26 = sub_B43CB0(a2);
  v46(&v60, v44, v26);
  if ( !a3 && (unsigned __int8)sub_DFB150((__int64)&v60) )
  {
    v57 = *(__int64 (__fastcall **)(_QWORD *, _QWORD, __int64 *, __int64, __int64, unsigned __int64, __int64, __int64, __int64, __int64, _QWORD, _QWORD))(*v25 + 1976LL);
    v36 = sub_33738A0(a1);
    v38 = v57(v25, *(_QWORD *)(a1 + 864), &v58, v36, v37, v42, v52, v54, v50, v51, v41, *((_QWORD *)&v41 + 1));
    v31 = *(_QWORD *)(a1 + 864);
    v32 = v38;
    v56 = v39;
    if ( v38 )
      goto LABEL_19;
  }
  else
  {
    v27 = *(_QWORD *)(a1 + 864);
    v28 = sub_33738A0(a1);
    v30 = sub_33F65D0(v27, v28, v29, (unsigned int)&v58, v50, v51, v52, v54, v40, v41, v47, v48, v42, 0, 0, a3);
    v31 = *(_QWORD *)(a1 + 864);
    v32 = v30;
    v56 = v33;
    if ( v30 )
    {
LABEL_19:
      v53 = v31;
      nullsub_1875(v32, v31, 0);
      *(_QWORD *)(v53 + 384) = v32;
      *(_DWORD *)(v53 + 392) = v56;
      sub_33E2B60(v53, 0);
      goto LABEL_20;
    }
  }
  *(_QWORD *)(v31 + 384) = 0;
  *(_DWORD *)(v31 + 392) = v56;
LABEL_20:
  v62 = a2;
  v34 = sub_337DC20(a1 + 8, &v62);
  *v34 = v32;
  *((_DWORD *)v34 + 2) = v56;
  sub_DFE7B0(&v60);
  if ( v58 )
    sub_B91220((__int64)&v58, v58);
}
