// Function: sub_339E310
// Address: 0x339e310
//
void __fastcall sub_339E310(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  __int64 v5; // rax
  int v6; // edx
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // r13
  unsigned __int16 v10; // ax
  unsigned __int8 v11; // dl
  __int64 v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rdx
  unsigned __int16 *v18; // rax
  __int64 v19; // r8
  __int64 v20; // rcx
  __int64 v21; // rdx
  int v22; // r9d
  __m128i v23; // xmm0
  __m128i v24; // xmm1
  __int64 *v25; // rax
  char v26; // al
  _QWORD *v27; // rax
  __int64 v28; // rax
  unsigned __int16 v29; // si
  _QWORD *v30; // rdi
  int v31; // eax
  __int64 v32; // rax
  unsigned __int64 v33; // r13
  _QWORD *v34; // r15
  __int64 v35; // rax
  __int64 v36; // r9
  __int64 v37; // r15
  int v38; // edx
  int v39; // r13d
  _QWORD *v40; // rax
  __int64 v41; // rax
  unsigned __int64 v42; // rcx
  unsigned __int8 v43; // al
  int v44; // edx
  __int64 v45; // rax
  __int64 v46; // r8
  _QWORD *v47; // rax
  __int128 v48; // [rsp+0h] [rbp-160h]
  __int64 v49; // [rsp+10h] [rbp-150h]
  _QWORD *v50; // [rsp+18h] [rbp-148h]
  char v51; // [rsp+27h] [rbp-139h]
  int v52; // [rsp+28h] [rbp-138h]
  int v53; // [rsp+28h] [rbp-138h]
  __int64 v54; // [rsp+28h] [rbp-138h]
  __int128 v55; // [rsp+30h] [rbp-130h]
  unsigned __int8 v56; // [rsp+40h] [rbp-120h]
  void (__fastcall *v57)(__int64 *, __int64, __int64); // [rsp+40h] [rbp-120h]
  __int64 v58; // [rsp+48h] [rbp-118h]
  __int64 v59; // [rsp+50h] [rbp-110h]
  __int64 v60; // [rsp+58h] [rbp-108h]
  __int128 v61; // [rsp+60h] [rbp-100h]
  __int64 v62; // [rsp+70h] [rbp-F0h]
  __int64 v63; // [rsp+78h] [rbp-E8h]
  __int64 v65; // [rsp+88h] [rbp-D8h]
  __int64 v66; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v67; // [rsp+A8h] [rbp-B8h] BYREF
  __int64 v68; // [rsp+B0h] [rbp-B0h] BYREF
  int v69; // [rsp+B8h] [rbp-A8h]
  __int128 v70; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 v71; // [rsp+D0h] [rbp-90h]
  __m128i v72; // [rsp+E0h] [rbp-80h] BYREF
  __m128i v73; // [rsp+F0h] [rbp-70h] BYREF
  __int64 v74; // [rsp+100h] [rbp-60h] BYREF
  unsigned __int64 v75; // [rsp+108h] [rbp-58h]
  __m128i v76; // [rsp+110h] [rbp-50h]
  __m128i v77; // [rsp+120h] [rbp-40h]

  v5 = *(_QWORD *)a1;
  v6 = *(_DWORD *)(a1 + 848);
  v68 = 0;
  v69 = v6;
  if ( v5 )
  {
    if ( &v68 != (__int64 *)(v5 + 48) )
    {
      v7 = *(_QWORD *)(v5 + 48);
      v68 = v7;
      if ( v7 )
        sub_B96E90((__int64)&v68, v7, 1);
    }
  }
  v8 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v9 = *(_QWORD *)(a2 - 32 * v8);
  if ( a3 )
  {
    v10 = sub_A74840((_QWORD *)(a2 + 72), 0);
    v11 = 0;
    if ( HIBYTE(v10) )
      v11 = v10;
    v56 = v11;
    v12 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    v13 = *(_QWORD *)(a2 + 32 * (1 - v12));
    v58 = *(_QWORD *)(a2 + 32 * (2 - v12));
  }
  else
  {
    v41 = *(_QWORD *)(a2 + 32 * (1 - v8));
    v42 = *(_QWORD *)(v41 + 24);
    if ( *(_DWORD *)(v41 + 32) > 0x40u )
      v42 = *(_QWORD *)v42;
    v43 = 0;
    if ( v42 )
    {
      _BitScanReverse64(&v42, v42);
      v43 = 63 - (v42 ^ 0x3F);
    }
    v56 = v43;
    v13 = *(_QWORD *)(a2 + 32 * (2 - v8));
    v58 = *(_QWORD *)(a2 + 32 * (3 - v8));
  }
  v62 = sub_338B750(a1, v9);
  v63 = v14;
  *(_QWORD *)&v61 = sub_338B750(a1, v58);
  *((_QWORD *)&v61 + 1) = v15;
  *(_QWORD *)&v55 = sub_338B750(a1, v13);
  v16 = *(_QWORD *)(a1 + 864);
  *((_QWORD *)&v55 + 1) = v17;
  v18 = (unsigned __int16 *)(*(_QWORD *)(v62 + 48) + 16LL * (unsigned int)v63);
  v19 = *((_QWORD *)v18 + 1);
  v20 = *v18;
  v74 = 0;
  LODWORD(v75) = 0;
  *(_QWORD *)&v48 = sub_33F17F0(v16, 51, &v74, v20, v19);
  *((_QWORD *)&v48 + 1) = v21;
  if ( v74 )
    sub_B91220((__int64)&v74, v74);
  v59 = *(unsigned __int16 *)(*(_QWORD *)(v61 + 48) + 16LL * DWORD2(v61));
  v60 = *(_QWORD *)(*(_QWORD *)(v61 + 48) + 16LL * DWORD2(v61) + 8);
  sub_B91FC0(v72.m128i_i64, a2);
  if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 && sub_B91C10(a2, 29) && (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
    v22 = sub_B91C10(a2, 4);
  else
    v22 = 0;
  v23 = _mm_load_si128(&v72);
  v24 = _mm_load_si128(&v73);
  v74 = v9;
  v75 = 0xBFFFFFFFFFFFFFFELL;
  v25 = *(__int64 **)(a1 + 872);
  v76 = v23;
  v77 = v24;
  if ( v25 && (v52 = v22, v26 = sub_CF4FA0(*v25, (__int64)&v74, (__int64)(v25 + 1), 0), v22 = v52, !v26) )
  {
    v27 = *(_QWORD **)(a1 + 864);
    v51 = 0;
    v49 = 0;
    v50 = v27 + 36;
    if ( (*(_BYTE *)(a2 + 7) & 0x20) == 0 )
      goto LABEL_37;
  }
  else
  {
    v27 = *(_QWORD **)(a1 + 864);
    v51 = 1;
    v50 = (_QWORD *)v27[48];
    v49 = v27[49];
    if ( (*(_BYTE *)(a2 + 7) & 0x20) == 0 )
      goto LABEL_37;
  }
  v53 = v22;
  v28 = sub_B91C10(a2, 9);
  v22 = v53;
  if ( v28 )
  {
    v27 = *(_QWORD **)(a1 + 864);
    v29 = 9;
    goto LABEL_20;
  }
  v27 = *(_QWORD **)(a1 + 864);
LABEL_37:
  v29 = 1;
LABEL_20:
  v30 = (_QWORD *)v27[5];
  BYTE4(v71) = 0;
  *((_QWORD *)&v70 + 1) = 0;
  *(_QWORD *)&v70 = v9 & 0xFFFFFFFFFFFFFFFBLL;
  v31 = 0;
  if ( v9 )
  {
    v32 = *(_QWORD *)(v9 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v32 + 8) - 17 <= 1 )
      v32 = **(_QWORD **)(v32 + 16);
    v31 = *(_DWORD *)(v32 + 8) >> 8;
  }
  LODWORD(v71) = v31;
  v33 = sub_2E7BD70(v30, v29, -1, v56, (int)&v72, v22, v70, v71, 1u, 0, 0);
  v34 = *(_QWORD **)(*(_QWORD *)(a1 + 864) + 16LL);
  v54 = v34[1];
  v57 = *(void (__fastcall **)(__int64 *, __int64, __int64))(*(_QWORD *)v54 + 104LL);
  v35 = sub_B43CB0(a2);
  v57(&v66, v54, v35);
  *(_QWORD *)&v70 = 0;
  DWORD2(v70) = 0;
  if ( !a3 && (unsigned __int8)sub_DFB150((__int64)&v66) )
  {
    v37 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD, __int64 *, _QWORD *, __int64, unsigned __int64, __int128 *, __int64, __int64, _QWORD, _QWORD, _QWORD, _QWORD))(*v34 + 1968LL))(
            v34,
            *(_QWORD *)(a1 + 864),
            &v68,
            v50,
            v49,
            v33,
            &v70,
            v62,
            v63,
            v61,
            *((_QWORD *)&v61 + 1),
            v55,
            *((_QWORD *)&v55 + 1));
    v39 = v44;
    if ( !v51 )
      goto LABEL_27;
  }
  else
  {
    v37 = sub_33E8F60(
            *(_QWORD *)(a1 + 864),
            v59,
            v60,
            (unsigned int)&v68,
            (_DWORD)v50,
            v49,
            v62,
            v63,
            v48,
            v55,
            v61,
            v59,
            v60,
            v33,
            0,
            0,
            a3);
    v39 = v38;
    *(_QWORD *)&v70 = v37;
    DWORD2(v70) = v38;
    if ( !v51 )
      goto LABEL_27;
  }
  v45 = *(unsigned int *)(a1 + 136);
  v46 = v70;
  if ( v45 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 140) )
  {
    v65 = v70;
    sub_C8D5F0(a1 + 128, (const void *)(a1 + 144), v45 + 1, 0x10u, v70, v36);
    v45 = *(unsigned int *)(a1 + 136);
    v46 = v65;
  }
  v47 = (_QWORD *)(*(_QWORD *)(a1 + 128) + 16 * v45);
  *v47 = v46;
  v47[1] = 1;
  ++*(_DWORD *)(a1 + 136);
LABEL_27:
  v67 = a2;
  v40 = sub_337DC20(a1 + 8, &v67);
  *v40 = v37;
  *((_DWORD *)v40 + 2) = v39;
  sub_DFE7B0(&v66);
  if ( v68 )
    sub_B91220((__int64)&v68, v68);
}
