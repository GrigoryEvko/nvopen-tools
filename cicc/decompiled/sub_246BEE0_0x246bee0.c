// Function: sub_246BEE0
// Address: 0x246bee0
//
void __fastcall sub_246BEE0(__int64 a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v4; // rsi
  __int64 v5; // rax
  unsigned __int64 v6; // rax
  char v7; // dl
  unsigned __int8 *v8; // r13
  __int64 v9; // rax
  __int64 v10; // r10
  __int64 v11; // r15
  unsigned int v12; // eax
  unsigned __int8 *v13; // r10
  __int64 (__fastcall *v14)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rax
  char *v18; // rax
  signed __int64 v19; // rdx
  _QWORD *v20; // rax
  bool v21; // zf
  __int64 v22; // rcx
  __int64 v23; // r15
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rdx
  unsigned __int64 v27; // rax
  char v28; // cl
  __int64 v29; // rax
  __int64 v30; // rax
  _QWORD *v31; // rdx
  __int64 v32; // r11
  _QWORD *v33; // rax
  _QWORD *v34; // rdx
  _QWORD *v35; // r10
  char *v36; // rax
  signed __int64 v37; // rdx
  __int64 v38; // rax
  unsigned int *v39; // r13
  __int64 v40; // rdx
  unsigned int v41; // esi
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // [rsp+8h] [rbp-178h]
  __int64 v45; // [rsp+10h] [rbp-170h]
  __int64 v46; // [rsp+10h] [rbp-170h]
  unsigned int v47; // [rsp+18h] [rbp-168h]
  unsigned __int8 *v48; // [rsp+18h] [rbp-168h]
  _QWORD *v49; // [rsp+18h] [rbp-168h]
  _QWORD *v50; // [rsp+18h] [rbp-168h]
  _QWORD *v51; // [rsp+18h] [rbp-168h]
  unsigned int *v52; // [rsp+18h] [rbp-168h]
  unsigned __int8 *v53; // [rsp+18h] [rbp-168h]
  _BYTE v54[32]; // [rsp+30h] [rbp-150h] BYREF
  __int16 v55; // [rsp+50h] [rbp-130h]
  __int64 v56; // [rsp+60h] [rbp-120h] BYREF
  unsigned __int8 *v57; // [rsp+68h] [rbp-118h]
  _QWORD *v58; // [rsp+70h] [rbp-110h]
  _QWORD *v59; // [rsp+78h] [rbp-108h]
  __int16 v60; // [rsp+80h] [rbp-100h]
  _BYTE v61[32]; // [rsp+90h] [rbp-F0h] BYREF
  __int16 v62; // [rsp+B0h] [rbp-D0h]
  unsigned int *v63; // [rsp+C0h] [rbp-C0h] BYREF
  int v64; // [rsp+C8h] [rbp-B8h]
  char v65; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v66; // [rsp+F8h] [rbp-88h]
  __int64 v67; // [rsp+100h] [rbp-80h]
  _QWORD *v68; // [rsp+108h] [rbp-78h]
  __int64 v69; // [rsp+110h] [rbp-70h]
  __int64 v70; // [rsp+118h] [rbp-68h]
  void *v71; // [rsp+140h] [rbp-40h]

  v4 = a3;
  if ( !a3 )
    v4 = (_QWORD *)a2;
  sub_2468350((__int64)&v63, v4);
  v5 = sub_B2BEC0(*(_QWORD *)a1);
  v6 = sub_BDB740(v5, *(_QWORD *)(a2 + 72));
  v8 = (unsigned __int8 *)sub_B33F60((__int64)&v63, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 80LL), v6, v7);
  if ( (unsigned __int8)sub_B4CE70(a2) )
  {
    v9 = *(_QWORD *)(a1 + 8);
    v55 = 257;
    v10 = *(_QWORD *)(a2 - 32);
    v60 = 257;
    v11 = *(_QWORD *)(v9 + 80);
    v45 = v10;
    v47 = sub_BCB060(*(_QWORD *)(v10 + 8));
    v12 = sub_BCB060(v11);
    v13 = (unsigned __int8 *)v45;
    if ( v47 < v12 )
    {
      v13 = (unsigned __int8 *)sub_A82F30(&v63, v45, v11, (__int64)v54, 0);
    }
    else if ( v47 > v12 )
    {
      v13 = (unsigned __int8 *)sub_A82DA0(&v63, v45, v11, (__int64)v54, 0, 0);
    }
    v14 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v69 + 32LL);
    if ( v14 == sub_9201A0 )
    {
      if ( *v8 > 0x15u || *v13 > 0x15u )
        goto LABEL_35;
      v48 = v13;
      if ( (unsigned __int8)sub_AC47B0(17) )
        v15 = sub_AD5570(17, (__int64)v8, v48, 0, 0);
      else
        v15 = sub_AABE40(0x11u, v8, v48);
      v13 = v48;
      v16 = v15;
    }
    else
    {
      v53 = v13;
      v42 = v14(v69, 17u, v8, v13, 0, 0);
      v13 = v53;
      v16 = v42;
    }
    if ( v16 )
    {
LABEL_14:
      v8 = (unsigned __int8 *)v16;
      goto LABEL_15;
    }
LABEL_35:
    v62 = 257;
    v16 = sub_B504D0(17, (__int64)v8, (__int64)v13, (__int64)v61, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)v70 + 16LL))(
      v70,
      v16,
      &v56,
      v66,
      v67);
    v39 = v63;
    v52 = &v63[4 * v64];
    if ( v63 != v52 )
    {
      do
      {
        v40 = *((_QWORD *)v39 + 1);
        v41 = *v39;
        v39 += 4;
        sub_B99FD0(v16, v41, v40);
      }
      while ( v52 != v39 );
    }
    goto LABEL_14;
  }
LABEL_15:
  v17 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)v17 )
  {
    v18 = (char *)sub_BD5D20(a2);
    v20 = sub_2461F00(*(__int64 **)(*(_QWORD *)a1 + 40LL), v18, v19);
    v21 = *(_BYTE *)(a1 + 634) == 0;
    v56 = a2;
    v62 = 257;
    v57 = v8;
    if ( v21 )
    {
      sub_921880(
        &v63,
        *(_QWORD *)(*(_QWORD *)(a1 + 8) + 480LL),
        *(_QWORD *)(*(_QWORD *)(a1 + 8) + 488LL),
        (int)&v56,
        2,
        (__int64)v61,
        0);
    }
    else
    {
      v58 = v20;
      sub_921880(
        &v63,
        *(_QWORD *)(*(_QWORD *)(a1 + 8) + 464LL),
        *(_QWORD *)(*(_QWORD *)(a1 + 8) + 472LL),
        (int)&v56,
        3,
        (__int64)v61,
        0);
    }
  }
  else
  {
    if ( *(_BYTE *)(a1 + 634) && byte_4FE8BC8 )
    {
      v56 = a2;
      v57 = v8;
      v62 = 257;
      sub_921880(&v63, *(_QWORD *)(v17 + 344), *(_QWORD *)(v17 + 352), (int)&v56, 2, (__int64)v61, 0);
    }
    else
    {
      v22 = sub_BCB2B0(v68);
      if ( **(_BYTE **)(a1 + 8) )
        v23 = (__int64)sub_2465B30((__int64 *)a1, a2, (__int64)&v63, v22, 1);
      else
        v23 = sub_2463FC0(a1, a2, &v63, 0x100u);
      v24 = 0;
      if ( *(_BYTE *)(a1 + 634) )
        v24 = (unsigned __int8)qword_4FE8AE8;
      v25 = sub_BCB2B0(v68);
      v26 = sub_ACD640(v25, v24, 0);
      _BitScanReverse64(&v27, 1LL << *(_WORD *)(a2 + 2));
      v28 = 63 - (v27 ^ 0x3F);
      LODWORD(v27) = 256;
      LOBYTE(v27) = v28;
      sub_B34240((__int64)&v63, v23, v26, (__int64)v8, v27, 0, 0, 0, 0);
    }
    if ( *(_BYTE *)(a1 + 634) && *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4LL) )
    {
      v29 = sub_BCB2D0(**(_QWORD ***)(*(_QWORD *)a1 + 40LL));
      v30 = sub_ACD640(v29, 0, 0);
      v31 = *(_QWORD **)(v30 + 8);
      v46 = v30;
      v32 = *(_QWORD *)(*(_QWORD *)a1 + 40LL);
      v62 = 257;
      v49 = v31;
      v44 = v32;
      BYTE4(v56) = 0;
      v33 = sub_BD2C40(88, unk_3F0FAE8);
      v34 = v49;
      v35 = v33;
      if ( v33 )
      {
        v50 = v33;
        sub_B30000((__int64)v33, v44, v34, 0, 8, v46, (__int64)v61, 0, 0, v56, 0);
        v35 = v50;
      }
      if ( byte_4FE8A08 )
      {
        v51 = v35;
        v36 = (char *)sub_BD5D20(a2);
        v59 = sub_2461F00(*(__int64 **)(*(_QWORD *)a1 + 40LL), v36, v37);
        v38 = *(_QWORD *)(a1 + 8);
        v62 = 257;
        v56 = a2;
        v57 = v8;
        v58 = v51;
        sub_921880(&v63, *(_QWORD *)(v38 + 312), *(_QWORD *)(v38 + 320), (int)&v56, 4, (__int64)v61, 0);
      }
      else
      {
        v56 = a2;
        v62 = 257;
        v43 = *(_QWORD *)(a1 + 8);
        v57 = v8;
        v58 = v35;
        sub_921880(&v63, *(_QWORD *)(v43 + 328), *(_QWORD *)(v43 + 336), (int)&v56, 3, (__int64)v61, 0);
      }
    }
  }
  nullsub_61();
  v71 = &unk_49DA100;
  nullsub_63();
  if ( v63 != (unsigned int *)&v65 )
    _libc_free((unsigned __int64)v63);
}
