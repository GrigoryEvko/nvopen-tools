// Function: sub_37A0500
// Address: 0x37a0500
//
unsigned __int8 *__fastcall sub_37A0500(__int64 *a1, unsigned __int64 a2, unsigned int a3, __m128i a4)
{
  unsigned __int16 *v6; // rax
  unsigned __int16 v7; // cx
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 (__fastcall *v13)(__int64, __int64, unsigned int, __int64); // r10
  __int64 v14; // rax
  unsigned __int16 v15; // si
  __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // r8
  int v20; // eax
  unsigned __int16 *v21; // rdx
  unsigned __int64 v22; // rbx
  unsigned int v23; // r15d
  int v24; // eax
  unsigned __int16 v25; // ax
  __int64 v26; // rsi
  __int16 v27; // ax
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rdx
  __int16 v32; // r10
  unsigned __int16 v33; // ax
  bool v34; // zf
  __int16 v35; // ax
  __int64 v36; // r8
  __int64 v37; // r9
  unsigned int v38; // edx
  __int16 v39; // r10
  __int64 v40; // rcx
  _QWORD *v41; // rbx
  __int64 v42; // rsi
  unsigned int v43; // esi
  __int64 v44; // r8
  unsigned int v45; // r9d
  unsigned __int8 *v46; // r14
  __int64 v48; // rdx
  unsigned __int64 v49; // rbx
  __int64 v50; // rdx
  __int64 v51; // rdx
  __int16 v52; // ax
  __int64 v53; // rdx
  __int64 v54; // rdx
  __int128 v55; // [rsp-10h] [rbp-100h]
  __int64 v56; // [rsp+8h] [rbp-E8h]
  __int64 v57; // [rsp+8h] [rbp-E8h]
  __int64 *v58; // [rsp+10h] [rbp-E0h]
  __int16 v59; // [rsp+20h] [rbp-D0h]
  __int16 v60; // [rsp+20h] [rbp-D0h]
  __int64 v62; // [rsp+28h] [rbp-C8h]
  __int64 v63; // [rsp+28h] [rbp-C8h]
  bool v64; // [rsp+30h] [rbp-C0h]
  unsigned int v65; // [rsp+30h] [rbp-C0h]
  __int16 v66; // [rsp+30h] [rbp-C0h]
  unsigned __int16 v67; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v68; // [rsp+58h] [rbp-98h]
  unsigned __int16 v69; // [rsp+60h] [rbp-90h] BYREF
  __int64 v70; // [rsp+68h] [rbp-88h]
  _QWORD v71[2]; // [rsp+70h] [rbp-80h] BYREF
  int v72; // [rsp+80h] [rbp-70h] BYREF
  __int64 v73; // [rsp+88h] [rbp-68h]
  __int64 v74; // [rsp+90h] [rbp-60h] BYREF
  int v75; // [rsp+98h] [rbp-58h]
  __int64 v76; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v77; // [rsp+A8h] [rbp-48h]
  __int64 v78; // [rsp+B0h] [rbp-40h]
  __int64 v79; // [rsp+B8h] [rbp-38h]

  v6 = *(unsigned __int16 **)(a2 + 48);
  v7 = *v6;
  v68 = *((_QWORD *)v6 + 1);
  v8 = *((_QWORD *)v6 + 3);
  LOWORD(v6) = v6[8];
  v67 = v7;
  v70 = v8;
  v69 = (unsigned __int16)v6;
  v9 = *(_QWORD *)(a1[1] + 64);
  v58 = (__int64 *)v9;
  v10 = sub_379AB60((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v11 = *a1;
  v71[0] = v10;
  v71[1] = v12;
  v13 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v11 + 592LL);
  v14 = *(_QWORD *)(a2 + 48) + 16LL * a3;
  v15 = *(_WORD *)v14;
  if ( v13 == sub_2D56A50 )
  {
    v16 = v15;
    v17 = v11;
    sub_2FE6CC0((__int64)&v76, v11, v9, v16, *(_QWORD *)(v14 + 8));
    LOWORD(v20) = v77;
    LOWORD(v72) = v77;
    v73 = v78;
  }
  else
  {
    v53 = v15;
    v17 = v9;
    v20 = v13(v11, v9, v53, *(_QWORD *)(v14 + 8));
    v72 = v20;
    v73 = v54;
  }
  if ( (_WORD)v20 )
  {
    v21 = word_4456340;
    v64 = (unsigned __int16)(v20 - 176) <= 0x34u;
    LOBYTE(v22) = v64;
    v23 = word_4456340[(unsigned __int16)v20 - 1];
    v24 = v67;
    if ( v67 )
    {
LABEL_5:
      v56 = 0;
      v25 = word_4456580[v24 - 1];
      goto LABEL_6;
    }
  }
  else
  {
    v49 = sub_3007240((__int64)&v72);
    v23 = v49;
    v24 = v67;
    v22 = HIDWORD(v49);
    v64 = v22;
    if ( v67 )
      goto LABEL_5;
  }
  v25 = sub_3009970((__int64)&v67, v17, (__int64)v21, v18, v19);
  v56 = v50;
LABEL_6:
  BYTE4(v76) = v22;
  v26 = v23;
  LODWORD(v76) = v23;
  v62 = v25;
  if ( v64 )
    v27 = sub_2D43AD0(v25, v23);
  else
    v27 = sub_2D43050(v25, v23);
  v31 = v62;
  v32 = v27;
  v63 = 0;
  if ( !v27 )
  {
    v26 = (unsigned int)v31;
    v52 = sub_3009450(v58, (unsigned int)v31, v56, v76, v29, v30);
    v63 = v31;
    v32 = v52;
  }
  if ( v69 )
  {
    v57 = 0;
    v33 = word_4456580[v69 - 1];
  }
  else
  {
    v60 = v32;
    v33 = sub_3009970((__int64)&v69, v26, v31, v28, v29);
    v32 = v60;
    v57 = v51;
  }
  LODWORD(v76) = v23;
  v34 = !v64;
  BYTE4(v76) = v22;
  v59 = v32;
  v65 = v33;
  if ( v34 )
  {
    v35 = sub_2D43050(v33, v23);
    v39 = v59;
    v38 = v65;
    v40 = 0;
    if ( v35 )
      goto LABEL_14;
  }
  else
  {
    v35 = sub_2D43AD0(v33, v23);
    v38 = v65;
    v39 = v59;
    v40 = 0;
    if ( v35 )
      goto LABEL_14;
  }
  v66 = v39;
  v35 = sub_3009450(v58, v38, v57, v76, v36, v37);
  v39 = v66;
  v40 = v48;
LABEL_14:
  LOWORD(v78) = v35;
  v41 = (_QWORD *)a1[1];
  v42 = *(_QWORD *)(a2 + 80);
  LOWORD(v76) = v39;
  v79 = v40;
  v77 = v63;
  v74 = v42;
  if ( v42 )
    sub_B96E90((__int64)&v74, v42, 1);
  *((_QWORD *)&v55 + 1) = 1;
  *(_QWORD *)&v55 = v71;
  v43 = *(_DWORD *)(a2 + 24);
  v75 = *(_DWORD *)(a2 + 72);
  v46 = sub_3411BE0(v41, v43, (__int64)&v74, (unsigned __int16 *)&v76, 2, 1, v55);
  if ( v74 )
    sub_B91220((__int64)&v74, v74);
  sub_378DDD0(a1, a2, (unsigned __int64)v46, a3, a4, v44, v45);
  return v46;
}
