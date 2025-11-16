// Function: sub_2C1D2D0
// Address: 0x2c1d2d0
//
__int64 __fastcall sub_2C1D2D0(__int64 a1, __int64 *a2)
{
  __int64 *v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rdi
  unsigned __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r14
  unsigned __int8 *v9; // rax
  __int64 v10; // r15
  unsigned __int8 v11; // al
  __int64 v12; // rcx
  unsigned int v13; // eax
  __int64 *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rsi
  _BYTE *v17; // rax
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // r12
  unsigned __int8 *v21; // rax
  int v22; // r14d
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // r14
  __int64 v28; // rax
  __int64 v29; // r12
  _QWORD *v30; // r15
  __int16 v31; // dx
  __int64 v32; // rsi
  unsigned __int8 v33; // cl
  char v34; // dl
  __int64 v35; // rax
  unsigned __int8 *v36; // rsi
  __int64 v37; // r14
  unsigned __int8 *v38; // rsi
  __int64 v40; // r15
  unsigned __int8 v41; // al
  __int64 v42; // rax
  unsigned int v43; // eax
  _BYTE *v44; // rax
  _BYTE *v45; // rax
  __int64 v46; // r14
  __int64 v47; // rax
  int v48; // eax
  __int64 v49; // rax
  __int64 v50; // rdx
  int v51; // [rsp+Ch] [rbp-174h]
  __int64 v52; // [rsp+10h] [rbp-170h]
  __int64 v53; // [rsp+18h] [rbp-168h]
  __int16 v54; // [rsp+22h] [rbp-15Eh]
  int v55; // [rsp+24h] [rbp-15Ch]
  _QWORD **v56; // [rsp+28h] [rbp-158h]
  __int64 v57; // [rsp+28h] [rbp-158h]
  __int64 v58; // [rsp+28h] [rbp-158h]
  unsigned int v59; // [rsp+28h] [rbp-158h]
  _BYTE *v60; // [rsp+28h] [rbp-158h]
  __int64 v61; // [rsp+30h] [rbp-150h]
  int v62; // [rsp+38h] [rbp-148h]
  __int16 v63; // [rsp+3Ch] [rbp-144h]
  unsigned __int8 v64; // [rsp+3Fh] [rbp-141h]
  __int64 v65; // [rsp+40h] [rbp-140h]
  __int64 v66; // [rsp+48h] [rbp-138h]
  __int64 v67; // [rsp+50h] [rbp-130h]
  __int64 v68; // [rsp+50h] [rbp-130h]
  __int64 v69; // [rsp+58h] [rbp-128h]
  __int64 v70; // [rsp+60h] [rbp-120h]
  int v71; // [rsp+60h] [rbp-120h]
  __int64 v72; // [rsp+68h] [rbp-118h]
  __int64 v73; // [rsp+68h] [rbp-118h]
  __int64 v76; // [rsp+88h] [rbp-F8h]
  unsigned int v77; // [rsp+90h] [rbp-F0h]
  _DWORD v78[8]; // [rsp+A0h] [rbp-E0h] BYREF
  __int16 v79; // [rsp+C0h] [rbp-C0h]
  __int64 v80[4]; // [rsp+D0h] [rbp-B0h] BYREF
  __int16 v81; // [rsp+F0h] [rbp-90h]
  const char *v82[2]; // [rsp+100h] [rbp-80h] BYREF
  char v83; // [rsp+110h] [rbp-70h] BYREF
  __int16 v84; // [rsp+120h] [rbp-60h]

  if ( !*(_DWORD *)(a1 + 56) )
    BUG();
  v2 = *(__int64 **)(a1 + 48);
  v3 = a2[113];
  v70 = *(_QWORD *)(a1 + 152);
  v4 = *(_QWORD *)(*v2 + 40);
  v69 = *(_QWORD *)(a1 + 160);
  if ( !v69 )
    v69 = *(_QWORD *)(a1 + 136);
  v62 = *(_DWORD *)(v3 + 104);
  v63 = *(_WORD *)(v3 + 108);
  v61 = *(_QWORD *)(v3 + 96);
  v64 = *(_BYTE *)(v3 + 110);
  v5 = *(_QWORD *)(v70 + 40);
  if ( v5 )
  {
    if ( (unsigned __int8)sub_920620(v5) )
      *(_DWORD *)(v3 + 104) = sub_B45210(*(_QWORD *)(v70 + 40));
    v2 = *(__int64 **)(a1 + 48);
  }
  LODWORD(v82[0]) = 0;
  BYTE4(v82[0]) = 0;
  v72 = sub_2BFB120((__int64)a2, v2[1], (unsigned int *)v82);
  v53 = *(_QWORD *)(v3 + 56);
  v54 = *(_WORD *)(v3 + 64);
  v66 = *(_QWORD *)(v3 + 48);
  v65 = sub_2BF3650((__int64)(a2 + 12), a1);
  v6 = sub_986580(v65);
  sub_D5F1F0(v3, v6);
  if ( *(_BYTE *)v69 == 67 )
  {
    v46 = *(_QWORD *)(v69 + 8);
    v84 = 257;
    v47 = sub_A82DA0((unsigned int **)v3, v72, v46, (__int64)v82, 0, 0);
    HIDWORD(v80[0]) = 0;
    v84 = 257;
    v72 = v47;
    v4 = sub_2C13D90(v3, 38, v4, v46, (__int64)v82, 0, LODWORD(v80[0]));
  }
  v84 = 257;
  v7 = sub_B37620((unsigned int **)v3, a2[1], v4, (__int64 *)v82);
  v8 = a2[113];
  v67 = v7;
  v9 = *(unsigned __int8 **)(v70 + 40);
  if ( v9 )
    v55 = *v9 - 29;
  else
    v55 = 31;
  v10 = *(_QWORD *)(v67 + 8);
  v11 = *(_BYTE *)(v10 + 8);
  v12 = v10;
  LODWORD(v76) = *(_DWORD *)(v10 + 32);
  BYTE4(v76) = v11 == 18;
  if ( (unsigned int)v11 - 17 <= 1 )
  {
    v12 = **(_QWORD **)(v10 + 16);
    v11 = *(_BYTE *)(v12 + 8);
  }
  v82[0] = &v83;
  v82[1] = (const char *)0x800000000LL;
  if ( v11 <= 3u || v11 == 5 || (v16 = v10, (v11 & 0xFD) == 4) )
  {
    v56 = (_QWORD **)v12;
    v13 = sub_BCB060(v12);
    v14 = (__int64 *)sub_BCCE00(*v56, v13);
    v15 = sub_BCE1B0(v14, v76);
    v12 = (__int64)v56;
    v16 = v15;
  }
  v57 = v12;
  v81 = 257;
  v17 = (_BYTE *)sub_B33FB0(v8, v16, (__int64)v80);
  if ( *(_BYTE *)(v57 + 8) == 12 )
  {
    v60 = v17;
    v81 = 257;
    v44 = (_BYTE *)sub_B37620((unsigned int **)v8, v76, v72, v80);
    v81 = 257;
    v45 = (_BYTE *)sub_A81850((unsigned int **)v8, v60, v44, (__int64)v80, 0, 0);
    v80[0] = (__int64)"induction";
    v81 = 259;
    v68 = sub_929C50((unsigned int **)v8, (_BYTE *)v67, v45, (__int64)v80, 0, 0);
  }
  else
  {
    v81 = 257;
    v58 = sub_A83320((unsigned int **)v8, v17, v10, (__int64)v80, 0);
    v81 = 257;
    v18 = sub_B37620((unsigned int **)v8, v76, v72, v80);
    v79 = 257;
    if ( *(_BYTE *)(v8 + 108) )
    {
      v19 = sub_B35400(v8, 0x6Cu, v58, v18, v77, (__int64)v78, 0, 0, 0);
    }
    else
    {
      v52 = v18;
      v19 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(v8 + 80) + 40LL))(
              *(_QWORD *)(v8 + 80),
              18,
              v58,
              v18,
              *(unsigned int *)(v8 + 104));
      if ( !v19 )
      {
        v48 = *(_DWORD *)(v8 + 104);
        v81 = 257;
        v51 = v48;
        v49 = sub_B504D0(18, v58, v52, (__int64)v80, 0, 0);
        v50 = *(_QWORD *)(v8 + 96);
        v19 = v49;
        if ( v50 )
          sub_B99FD0(v49, 3u, v50);
        sub_B45150(v19, v51);
        (*(void (__fastcall **)(_QWORD, __int64, _DWORD *, _QWORD, _QWORD))(**(_QWORD **)(v8 + 88) + 16LL))(
          *(_QWORD *)(v8 + 88),
          v19,
          v78,
          *(_QWORD *)(v8 + 56),
          *(_QWORD *)(v8 + 64));
        sub_94AAF0((unsigned int **)v8, v19);
      }
    }
    v78[1] = 0;
    v80[0] = (__int64)"induction";
    v81 = 259;
    v68 = sub_2C137C0(v8, v55, v67, v19, v78[0], (__int64)v80, 0);
  }
  v20 = *(_QWORD *)(v72 + 8);
  if ( *(_BYTE *)(v20 + 8) == 12 )
  {
    v71 = 13;
    v22 = 17;
  }
  else
  {
    v21 = *(unsigned __int8 **)(v70 + 40);
    if ( v21 )
    {
      v22 = 18;
      v71 = *v21 - 29;
    }
    else
    {
      v71 = 31;
      v22 = 18;
    }
  }
  v23 = *(_QWORD *)(a1 + 48);
  if ( *(_DWORD *)(a1 + 56) == 5 && (v24 = *(_QWORD *)(v23 + 24)) != 0 )
  {
    v25 = sub_2BFB640((__int64)a2, v24, 0);
    v26 = v66;
    v27 = v25;
    if ( v66 )
    {
LABEL_26:
      sub_A88F30(v3, v26, v53, v54);
      goto LABEL_27;
    }
  }
  else
  {
    BYTE4(v82[0]) = 0;
    LODWORD(v82[0]) = 0;
    v40 = sub_2BFB120((__int64)a2, *(_QWORD *)(v23 + 16), (unsigned int *)v82);
    v41 = *(_BYTE *)(*(_QWORD *)(v72 + 8) + 8LL);
    if ( v41 <= 3u || v41 == 5 || (v41 & 0xFD) == 4 )
    {
      v84 = 257;
      v40 = sub_A83320((unsigned int **)v3, (_BYTE *)v40, v20, (__int64)v82, 0);
    }
    else
    {
      v84 = 257;
      v59 = sub_BCB060(*(_QWORD *)(v40 + 8));
      v43 = sub_BCB060(v20);
      if ( v59 < v43 )
      {
        v40 = sub_A82F30((unsigned int **)v3, v40, v20, (__int64)v82, 0);
      }
      else if ( v59 > v43 )
      {
        v40 = sub_A82DA0((unsigned int **)v3, v40, v20, (__int64)v82, 0, 0);
      }
    }
    HIDWORD(v80[0]) = 0;
    v84 = 257;
    v42 = sub_2C137C0(v3, v22, v72, v40, LODWORD(v80[0]), (__int64)v82, 0);
    v84 = 257;
    v26 = v66;
    v27 = sub_B37620((unsigned int **)v3, a2[1], v42, (__int64 *)v82);
    if ( v66 )
      goto LABEL_26;
  }
  *(_QWORD *)(v3 + 48) = 0;
  *(_QWORD *)(v3 + 56) = 0;
  *(_WORD *)(v3 + 64) = 0;
LABEL_27:
  v82[0] = "vec.ind";
  v84 = 259;
  v73 = *(_QWORD *)(v68 + 8);
  v28 = sub_BD2DA0(80);
  v29 = v28;
  if ( v28 )
  {
    v30 = (_QWORD *)v28;
    sub_B44260(v28, v73, 55, 0x8000000u, 0, 0);
    *(_DWORD *)(v29 + 72) = 2;
    sub_BD6B50((unsigned __int8 *)v29, v82);
    sub_BD2A10(v29, *(_DWORD *)(v29 + 72), 1);
  }
  else
  {
    v30 = 0;
  }
  v32 = sub_AA5190(a2[13]);
  if ( v32 )
  {
    v33 = v31;
    v34 = HIBYTE(v31);
  }
  else
  {
    v34 = 0;
    v33 = 0;
  }
  v35 = v33;
  BYTE1(v35) = v34;
  sub_B44220(v30, v32, v35);
  v82[0] = *(const char **)(a1 + 88);
  if ( v82[0] )
    sub_2AAAFA0((__int64 *)v82);
  if ( (const char **)(v29 + 48) != v82 )
  {
    sub_9C6650((_QWORD *)(v29 + 48));
    v36 = (unsigned __int8 *)v82[0];
    *(const char **)(v29 + 48) = v82[0];
    if ( v36 )
    {
      sub_B976B0((__int64)v82, v36, v29 + 48);
      v82[0] = 0;
    }
  }
  sub_9C6650(v82);
  sub_2BF26E0((__int64)a2, a1 + 96, v29, 0);
  HIDWORD(v80[0]) = 0;
  v82[0] = "vec.ind.next";
  v84 = 259;
  v37 = sub_2C137C0(v3, v71, v29, v27, LODWORD(v80[0]), (__int64)v82, 0);
  if ( *(_BYTE *)v69 == 67 )
    sub_2BF08A0((__int64)a2, (_BYTE *)v37, (_BYTE *)v69);
  v82[0] = *(const char **)(a1 + 88);
  if ( v82[0] )
    sub_2AAAFA0((__int64 *)v82);
  if ( (const char **)(v37 + 48) != v82 )
  {
    sub_9C6650((_QWORD *)(v37 + 48));
    v38 = (unsigned __int8 *)v82[0];
    *(const char **)(v37 + 48) = v82[0];
    if ( v38 )
    {
      sub_B976B0((__int64)v82, v38, v37 + 48);
      v82[0] = 0;
    }
  }
  sub_9C6650(v82);
  sub_F0A850(v29, v68, v65);
  sub_F0A850(v29, v37, v65);
  *(_QWORD *)(v3 + 96) = v61;
  *(_DWORD *)(v3 + 104) = v62;
  *(_WORD *)(v3 + 108) = v63;
  *(_BYTE *)(v3 + 110) = v64;
  return v64;
}
