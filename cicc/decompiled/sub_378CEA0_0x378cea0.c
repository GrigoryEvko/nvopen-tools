// Function: sub_378CEA0
// Address: 0x378cea0
//
__int64 __fastcall sub_378CEA0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rsi
  unsigned __int64 *v5; // rdx
  __int16 *v6; // rax
  unsigned __int16 v7; // bx
  __int64 v8; // rbx
  unsigned __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // r14
  __int128 v12; // rax
  __int64 v13; // r9
  __int128 v14; // rax
  __int64 v15; // r9
  __int64 v16; // rax
  unsigned __int16 v17; // ax
  __int64 v18; // rdx
  __int128 v19; // rax
  __int64 v20; // rax
  unsigned int v21; // edx
  __int64 v22; // r9
  __int128 v23; // rax
  _QWORD *v24; // r13
  __int64 v25; // r9
  unsigned __int8 *v26; // r8
  __int64 v27; // rdx
  __int64 v28; // r9
  unsigned int v29; // ecx
  __int64 v30; // rdx
  __int16 v31; // ax
  __int64 v32; // rdx
  unsigned int v33; // esi
  __int64 v34; // r14
  bool v36; // al
  __int128 v37; // [rsp-10h] [rbp-120h]
  __int64 v38; // [rsp+10h] [rbp-100h]
  __int64 (__fastcall *v39)(__int64, __int64, __int64, _QWORD, __int64); // [rsp+18h] [rbp-F8h]
  __int64 v40; // [rsp+18h] [rbp-F8h]
  __int64 v41; // [rsp+20h] [rbp-F0h]
  __int64 v42; // [rsp+20h] [rbp-F0h]
  _QWORD *v43; // [rsp+28h] [rbp-E8h]
  unsigned int v44; // [rsp+28h] [rbp-E8h]
  __int128 v45; // [rsp+30h] [rbp-E0h]
  __int128 v46; // [rsp+40h] [rbp-D0h]
  unsigned int v47; // [rsp+40h] [rbp-D0h]
  unsigned __int16 v48; // [rsp+50h] [rbp-C0h]
  unsigned __int8 *v49; // [rsp+50h] [rbp-C0h]
  __int64 v50; // [rsp+58h] [rbp-B8h]
  __int64 v51; // [rsp+60h] [rbp-B0h] BYREF
  int v52; // [rsp+68h] [rbp-A8h]
  __int128 v53; // [rsp+70h] [rbp-A0h] BYREF
  __int128 v54; // [rsp+80h] [rbp-90h] BYREF
  __int16 v55; // [rsp+90h] [rbp-80h] BYREF
  __int64 v56; // [rsp+98h] [rbp-78h]
  __m128i v57[2]; // [rsp+A0h] [rbp-70h] BYREF
  __m128i v58[5]; // [rsp+C0h] [rbp-50h] BYREF

  v4 = *(_QWORD *)(a2 + 80);
  v51 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v51, v4, 1);
  v5 = *(unsigned __int64 **)(a2 + 40);
  v52 = *(_DWORD *)(a2 + 72);
  v6 = *(__int16 **)(a2 + 48);
  v7 = *v6;
  *(_QWORD *)&v53 = 0;
  *(_QWORD *)&v54 = 0;
  v48 = v7;
  v8 = *((_QWORD *)v6 + 1);
  DWORD2(v53) = 0;
  DWORD2(v54) = 0;
  v9 = *v5;
  v10 = v5[1];
  v11 = 16LL * (unsigned int)v10;
  sub_375E8D0((__int64)a1, v9, v10, (__int64)&v53, (__int64)&v54);
  sub_3777990(v57, a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL), a3);
  sub_3408380(
    v58,
    (_QWORD *)a1[1],
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL),
    *(unsigned __int16 *)(*(_QWORD *)(v9 + 48) + v11),
    *(_QWORD *)(*(_QWORD *)(v9 + 48) + v11 + 8),
    a3,
    (__int64)&v51);
  *(_QWORD *)&v12 = sub_33FB310(a1[1], v58[0].m128i_i64[0], v58[0].m128i_u32[2], (__int64)&v51, v48, v8, a3);
  v46 = v12;
  *(_QWORD *)&v14 = sub_340F900(
                      (_QWORD *)a1[1],
                      0x1A4u,
                      (__int64)&v51,
                      v48,
                      v8,
                      v13,
                      v53,
                      *(_OWORD *)v57,
                      *(_OWORD *)v58);
  v15 = a1[1];
  v45 = v14;
  v43 = (_QWORD *)v15;
  v38 = *a1;
  v39 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)*a1 + 528LL);
  v41 = *(_QWORD *)(v15 + 64);
  v16 = sub_2E79000(*(__int64 **)(v15 + 40));
  v17 = v39(v38, v16, v41, v48, v8);
  v40 = v18;
  LODWORD(v41) = v17;
  *(_QWORD *)&v19 = sub_33ED040(v43, 0x16u);
  v20 = sub_340F900(v43, 0xD0u, (__int64)&v51, v41, v40, (__int64)v43, v45, v46, v19);
  v44 = v21;
  v42 = v20;
  *(_QWORD *)&v23 = sub_340F900(
                      (_QWORD *)a1[1],
                      *(_DWORD *)(a2 + 24),
                      (__int64)&v51,
                      v48,
                      v8,
                      v22,
                      v54,
                      *(_OWORD *)&v57[1],
                      *(_OWORD *)&v58[1]);
  v24 = (_QWORD *)a1[1];
  v26 = sub_3406EB0(v24, 0x38u, (__int64)&v51, v48, v8, v25, v46, v23);
  v28 = v27;
  v29 = v48;
  v30 = *(_QWORD *)(v42 + 48) + 16LL * v44;
  v31 = *(_WORD *)v30;
  v32 = *(_QWORD *)(v30 + 8);
  v55 = v31;
  v56 = v32;
  if ( v31 )
  {
    v33 = ((unsigned __int16)(v31 - 17) < 0xD4u) + 205;
  }
  else
  {
    v47 = v48;
    v49 = v26;
    v50 = v28;
    v36 = sub_30070B0((__int64)&v55);
    v29 = v47;
    v26 = v49;
    v28 = v50;
    v33 = 205 - (!v36 - 1);
  }
  *((_QWORD *)&v37 + 1) = v28;
  *(_QWORD *)&v37 = v26;
  v34 = sub_340EC60(v24, v33, (__int64)&v51, v29, v8, 0, v42, v44, v45, v37);
  if ( v51 )
    sub_B91220((__int64)&v51, v51);
  return v34;
}
