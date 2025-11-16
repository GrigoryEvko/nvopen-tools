// Function: sub_37AA430
// Address: 0x37aa430
//
unsigned __int8 *__fastcall sub_37AA430(_QWORD *a1, __int64 a2, __m128i a3)
{
  unsigned int v3; // r15d
  __int64 v6; // rdx
  __int128 v7; // rax
  __int64 v8; // rsi
  _DWORD *v9; // r14
  __int16 *v10; // rax
  __int16 v11; // dx
  unsigned __int16 *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rsi
  __int16 v15; // ax
  __int64 v16; // rdx
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rax
  char v19; // cl
  __int64 *v20; // rdi
  __int16 v21; // ax
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rcx
  __int64 v25; // rsi
  _QWORD *v26; // r9
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // rdx
  unsigned __int16 *v30; // rdx
  __int64 v31; // rcx
  unsigned __int64 v32; // rsi
  int v33; // eax
  unsigned __int16 v34; // ax
  __int64 *v35; // rdi
  int v36; // eax
  __int64 v37; // r9
  unsigned int v38; // edx
  __int64 v39; // r8
  __int128 v40; // rax
  __int64 v41; // r9
  _DWORD *v42; // r15
  int v43; // edx
  int v44; // r9d
  __int64 v45; // rdx
  __int16 v46; // ax
  __int64 v47; // rdx
  unsigned __int16 v48; // cx
  unsigned int v49; // eax
  char v50; // r12
  bool v51; // al
  unsigned __int8 *v52; // r14
  __int64 v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // rdx
  __int64 v60; // [rsp+0h] [rbp-C0h]
  __int64 v61; // [rsp+8h] [rbp-B8h]
  char v62; // [rsp+8h] [rbp-B8h]
  __int64 (__fastcall *v63)(_DWORD *, __int64, __int64, __int64, __int64); // [rsp+10h] [rbp-B0h]
  __int64 v64; // [rsp+10h] [rbp-B0h]
  unsigned int v65; // [rsp+10h] [rbp-B0h]
  __int64 v66; // [rsp+18h] [rbp-A8h]
  _QWORD *v67; // [rsp+18h] [rbp-A8h]
  __int64 v68; // [rsp+18h] [rbp-A8h]
  __int64 v69; // [rsp+18h] [rbp-A8h]
  __int128 v70; // [rsp+20h] [rbp-A0h]
  _QWORD *v71; // [rsp+20h] [rbp-A0h]
  char v72; // [rsp+20h] [rbp-A0h]
  __int128 v73; // [rsp+30h] [rbp-90h]
  __int128 v74; // [rsp+30h] [rbp-90h]
  int v75; // [rsp+38h] [rbp-88h]
  __int64 v76; // [rsp+50h] [rbp-70h] BYREF
  int v77; // [rsp+58h] [rbp-68h]
  unsigned int v78; // [rsp+60h] [rbp-60h] BYREF
  __int64 v79; // [rsp+68h] [rbp-58h]
  unsigned int v80; // [rsp+70h] [rbp-50h] BYREF
  __int64 v81; // [rsp+78h] [rbp-48h]
  __int64 v82; // [rsp+80h] [rbp-40h] BYREF
  __int64 v83; // [rsp+88h] [rbp-38h]

  *(_QWORD *)&v73 = sub_379AB60((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  *((_QWORD *)&v73 + 1) = v6;
  *(_QWORD *)&v7 = sub_379AB60(
                     (__int64)a1,
                     *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                     *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v8 = *(_QWORD *)(a2 + 80);
  v70 = v7;
  v76 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v76, v8, 1);
  v9 = (_DWORD *)*a1;
  v77 = *(_DWORD *)(a2 + 72);
  v10 = *(__int16 **)(a2 + 48);
  v11 = *v10;
  v79 = *((_QWORD *)v10 + 1);
  v12 = (unsigned __int16 *)(*(_QWORD *)(v73 + 48) + 16LL * DWORD2(v73));
  v13 = a1[1];
  LOWORD(v78) = v11;
  v60 = *((_QWORD *)v12 + 1);
  v61 = *v12;
  v66 = *(_QWORD *)(v13 + 64);
  v63 = *(__int64 (__fastcall **)(_DWORD *, __int64, __int64, __int64, __int64))(*(_QWORD *)v9 + 528LL);
  v14 = sub_2E79000(*(__int64 **)(v13 + 40));
  LOWORD(v80) = v63(v9, v14, v66, v61, v60);
  v15 = v78;
  v81 = v16;
  if ( (_WORD)v78 )
  {
    if ( (unsigned __int16)(v78 - 17) <= 0xD3u )
      v15 = word_4456580[(unsigned __int16)v78 - 1];
LABEL_6:
    if ( v15 != 2 )
      goto LABEL_12;
    if ( (_WORD)v80 )
    {
      v19 = (unsigned __int16)(v80 - 176) <= 0x34u;
      LODWORD(v17) = word_4456340[(unsigned __int16)v80 - 1];
      LOBYTE(v18) = v19;
    }
    else
    {
      v17 = sub_3007240((__int64)&v80);
      v18 = HIDWORD(v17);
      v19 = BYTE4(v17);
    }
    v20 = *(__int64 **)(a1[1] + 64LL);
    LODWORD(v82) = v17;
    BYTE4(v82) = v18;
    if ( v19 )
    {
      v21 = sub_2D43AD0(2, v17);
      v24 = 0;
      if ( v21 )
      {
LABEL_11:
        LOWORD(v80) = v21;
        v81 = v24;
        goto LABEL_12;
      }
    }
    else
    {
      v21 = sub_2D43050(2, v17);
      v24 = 0;
      if ( v21 )
        goto LABEL_11;
    }
    v21 = sub_3009450(v20, 2, 0, v82, v22, v23);
    v24 = v59;
    goto LABEL_11;
  }
  if ( sub_30070B0((__int64)&v78) )
  {
    v15 = sub_3009970((__int64)&v78, v14, v56, v57, v58);
    goto LABEL_6;
  }
LABEL_12:
  v25 = *(_QWORD *)(a2 + 80);
  v26 = (_QWORD *)a1[1];
  v27 = *(_QWORD *)(a2 + 40);
  v82 = v25;
  if ( v25 )
  {
    v64 = v27;
    v67 = v26;
    sub_B96E90((__int64)&v82, v25, 1);
    v27 = v64;
    v26 = v67;
  }
  LODWORD(v83) = *(_DWORD *)(a2 + 72);
  *(_QWORD *)&v74 = sub_340F900(v26, 0xD0u, (__int64)&v82, v80, v81, (__int64)v26, v73, v70, *(_OWORD *)(v27 + 80));
  *((_QWORD *)&v74 + 1) = v29;
  if ( v82 )
    sub_B91220((__int64)&v82, v82);
  if ( (_WORD)v78 )
  {
    v30 = word_4456340;
    LOBYTE(v28) = (unsigned __int16)(v78 - 176) <= 0x34u;
    v31 = (unsigned int)v28;
    v32 = word_4456340[(unsigned __int16)v78 - 1];
    v33 = (unsigned __int16)v80;
    if ( (_WORD)v80 )
    {
LABEL_18:
      v68 = 0;
      v34 = word_4456580[v33 - 1];
      goto LABEL_19;
    }
  }
  else
  {
    v32 = sub_3007240((__int64)&v78);
    v33 = (unsigned __int16)v80;
    v31 = HIDWORD(v32);
    v28 = HIDWORD(v32);
    if ( (_WORD)v80 )
      goto LABEL_18;
  }
  v62 = v31;
  v72 = v28;
  v34 = sub_3009970((__int64)&v80, v32, (__int64)v30, v31, v28);
  LOBYTE(v31) = v62;
  v68 = v55;
  LOBYTE(v28) = v72;
LABEL_19:
  v35 = *(__int64 **)(a1[1] + 64LL);
  LODWORD(v82) = v32;
  BYTE4(v82) = v31;
  v65 = v34;
  if ( (_BYTE)v28 )
  {
    LOWORD(v36) = sub_2D43AD0(v34, v32);
    v38 = v65;
    v39 = 0;
    if ( (_WORD)v36 )
      goto LABEL_21;
  }
  else
  {
    LOWORD(v36) = sub_2D43050(v34, v32);
    v38 = v65;
    v39 = 0;
    if ( (_WORD)v36 )
      goto LABEL_21;
  }
  v36 = sub_3009450(v35, v38, v68, v82, 0, v37);
  HIWORD(v3) = HIWORD(v36);
  v39 = v54;
LABEL_21:
  LOWORD(v3) = v36;
  v69 = v39;
  v71 = (_QWORD *)a1[1];
  *(_QWORD *)&v40 = sub_3400EE0((__int64)v71, 0, (__int64)&v76, 0, a3);
  sub_3406EB0(v71, 0xA1u, (__int64)&v76, v3, v69, v41, v74, v40);
  v42 = (_DWORD *)*a1;
  v44 = v43;
  v45 = *(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL);
  v46 = *(_WORD *)v45;
  v47 = *(_QWORD *)(v45 + 8);
  LOWORD(v82) = v46;
  v83 = v47;
  if ( v46 )
  {
    v48 = v46 - 17;
    if ( (unsigned __int16)(v46 - 10) > 6u && (unsigned __int16)(v46 - 126) > 0x31u )
    {
      if ( v48 <= 0xD3u )
      {
LABEL_25:
        v49 = v42[17];
        goto LABEL_29;
      }
LABEL_28:
      v49 = v42[15];
      goto LABEL_29;
    }
    if ( v48 <= 0xD3u )
      goto LABEL_25;
  }
  else
  {
    v75 = v44;
    v50 = sub_3007030((__int64)&v82);
    v51 = sub_30070B0((__int64)&v82);
    v44 = v75;
    if ( v51 )
      goto LABEL_25;
    if ( !v50 )
      goto LABEL_28;
  }
  v49 = v42[16];
LABEL_29:
  if ( v49 > 2 )
    BUG();
  v52 = sub_33FAF80(a1[1], 215 - v49, (__int64)&v76, v78, v79, v44, a3);
  if ( v76 )
    sub_B91220((__int64)&v76, v76);
  return v52;
}
