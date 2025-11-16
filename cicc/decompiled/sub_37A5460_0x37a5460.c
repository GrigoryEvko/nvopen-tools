// Function: sub_37A5460
// Address: 0x37a5460
//
unsigned __int8 *__fastcall sub_37A5460(__int64 *a1, __int64 a2, __m128i a3)
{
  unsigned int v3; // r14d
  __int64 v6; // rsi
  __int16 *v7; // rax
  __int16 v8; // dx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // r15
  __int64 v13; // rdx
  unsigned __int16 *v14; // rax
  __int64 v15; // rsi
  __int16 v16; // ax
  __int64 v17; // rdx
  int v18; // ecx
  __int16 v19; // ax
  __int64 v20; // rsi
  _QWORD *v21; // rdi
  int v22; // r9d
  __int64 v23; // r8
  __int64 v24; // rdx
  unsigned __int16 *v25; // rdx
  __int64 v26; // rcx
  int v27; // eax
  unsigned __int16 v28; // ax
  __int64 v29; // rdx
  int v30; // eax
  __int64 v31; // r8
  __int128 v32; // rax
  __int64 v33; // r9
  _DWORD *v34; // r14
  int v35; // edx
  int v36; // r9d
  __int64 v37; // rdx
  __int16 v38; // ax
  __int64 v39; // rdx
  unsigned __int16 v40; // cx
  unsigned int v41; // eax
  char v42; // r12
  bool v43; // al
  unsigned __int8 *v44; // r14
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // rdx
  __int64 v50; // rdx
  __int64 v51; // [rsp-8h] [rbp-D8h]
  __int64 v52; // [rsp+8h] [rbp-C8h]
  __int64 v53; // [rsp+10h] [rbp-C0h]
  __int64 v54; // [rsp+18h] [rbp-B8h]
  __int64 (__fastcall *v55)(__int64, __int64, __int64, __int64, __int64); // [rsp+20h] [rbp-B0h]
  __int64 *v56; // [rsp+20h] [rbp-B0h]
  __int64 *v57; // [rsp+20h] [rbp-B0h]
  __int64 v58; // [rsp+28h] [rbp-A8h]
  unsigned int v59; // [rsp+28h] [rbp-A8h]
  __int64 v60; // [rsp+28h] [rbp-A8h]
  __int64 v61; // [rsp+30h] [rbp-A0h]
  unsigned int v62; // [rsp+30h] [rbp-A0h]
  __int64 v63; // [rsp+30h] [rbp-A0h]
  __int64 v64; // [rsp+38h] [rbp-98h]
  int v65; // [rsp+38h] [rbp-98h]
  unsigned int v66; // [rsp+38h] [rbp-98h]
  _QWORD *v67; // [rsp+38h] [rbp-98h]
  int v68; // [rsp+40h] [rbp-90h]
  __int128 v69; // [rsp+40h] [rbp-90h]
  int v70; // [rsp+48h] [rbp-88h]
  __int64 v71; // [rsp+50h] [rbp-80h] BYREF
  int v72; // [rsp+58h] [rbp-78h]
  unsigned int v73; // [rsp+60h] [rbp-70h] BYREF
  __int64 v74; // [rsp+68h] [rbp-68h]
  unsigned int v75; // [rsp+70h] [rbp-60h] BYREF
  __int64 v76; // [rsp+78h] [rbp-58h]
  __int64 v77; // [rsp+80h] [rbp-50h] BYREF
  __int64 v78; // [rsp+88h] [rbp-48h]
  __int64 v79; // [rsp+90h] [rbp-40h]
  int v80; // [rsp+98h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 80);
  v71 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v71, v6, 1);
  v72 = *(_DWORD *)(a2 + 72);
  v7 = *(__int16 **)(a2 + 48);
  v8 = *v7;
  v74 = *((_QWORD *)v7 + 1);
  v9 = *(_QWORD *)(a2 + 40);
  LOWORD(v73) = v8;
  v64 = *(_QWORD *)(v9 + 40);
  v68 = *(_DWORD *)(v9 + 48);
  v10 = sub_379AB60((__int64)a1, *(_QWORD *)v9, *(_QWORD *)(v9 + 8));
  v11 = a1[1];
  v12 = v10;
  v61 = v13;
  v14 = (unsigned __int16 *)(*(_QWORD *)(v10 + 48) + 16LL * (unsigned int)v13);
  v54 = *a1;
  v58 = *(_QWORD *)(v11 + 64);
  v52 = *((_QWORD *)v14 + 1);
  v53 = *v14;
  v55 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 528LL);
  v15 = sub_2E79000(*(__int64 **)(v11 + 40));
  LOWORD(v75) = v55(v54, v15, v58, v53, v52);
  v16 = v73;
  v76 = v17;
  if ( (_WORD)v73 )
  {
    if ( (unsigned __int16)(v73 - 17) <= 0xD3u )
      v16 = word_4456580[(unsigned __int16)v73 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v73) )
      goto LABEL_15;
    v16 = sub_3009970((__int64)&v73, v15, v46, v47, v48);
  }
  if ( v16 != 2 )
    goto LABEL_15;
  if ( !(_WORD)v75 )
  {
    if ( !sub_3007100((__int64)&v75) )
      goto LABEL_11;
    goto LABEL_42;
  }
  if ( (unsigned __int16)(v75 - 176) <= 0x34u )
  {
LABEL_42:
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( (_WORD)v75 )
    {
      if ( (unsigned __int16)(v75 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      goto LABEL_9;
    }
LABEL_11:
    v18 = sub_3007130((__int64)&v75, v15);
    goto LABEL_12;
  }
LABEL_9:
  v18 = word_4456340[(unsigned __int16)v75 - 1];
LABEL_12:
  v59 = v18;
  v56 = *(__int64 **)(a1[1] + 64);
  v19 = sub_2D43050(2, v18);
  v20 = 0;
  if ( !v19 )
  {
    v19 = sub_3009400(v56, 2, 0, v59, 0);
    v20 = v50;
  }
  LOWORD(v75) = v19;
  v76 = v20;
LABEL_15:
  v21 = (_QWORD *)a1[1];
  v77 = v12;
  v22 = *(_DWORD *)(a2 + 28);
  v78 = v61;
  v79 = v64;
  v80 = v68;
  *(_QWORD *)&v69 = sub_33FBA10(v21, 155, (__int64)&v71, v75, v76, v22, (__int64)&v77, 2);
  *((_QWORD *)&v69 + 1) = v24;
  if ( (_WORD)v73 )
  {
    if ( (unsigned __int16)(v73 - 176) > 0x34u )
      goto LABEL_17;
  }
  else if ( !sub_3007100((__int64)&v73) )
  {
    goto LABEL_20;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( !(_WORD)v73 )
  {
LABEL_20:
    v26 = (unsigned int)sub_3007130((__int64)&v73, v51);
    v27 = (unsigned __int16)v75;
    if ( !(_WORD)v75 )
      goto LABEL_18;
    goto LABEL_21;
  }
  if ( (unsigned __int16)(v73 - 176) <= 0x34u )
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
LABEL_17:
  v25 = word_4456340;
  v26 = word_4456340[(unsigned __int16)v73 - 1];
  v27 = (unsigned __int16)v75;
  if ( !(_WORD)v75 )
  {
LABEL_18:
    v65 = v26;
    v28 = sub_3009970((__int64)&v75, v51, (__int64)v25, v26, v23);
    LODWORD(v26) = v65;
    v60 = v29;
    goto LABEL_22;
  }
LABEL_21:
  v60 = 0;
  v28 = word_4456580[v27 - 1];
LABEL_22:
  v62 = v26;
  v57 = *(__int64 **)(a1[1] + 64);
  v66 = v28;
  LOWORD(v30) = sub_2D43050(v28, v26);
  v31 = 0;
  if ( !(_WORD)v30 )
  {
    v30 = sub_3009400(v57, v66, v60, v62, 0);
    HIWORD(v3) = HIWORD(v30);
    v31 = v49;
  }
  LOWORD(v3) = v30;
  v63 = v31;
  v67 = (_QWORD *)a1[1];
  *(_QWORD *)&v32 = sub_3400EE0((__int64)v67, 0, (__int64)&v71, 0, a3);
  sub_3406EB0(v67, 0xA1u, (__int64)&v71, v3, v63, v33, v69, v32);
  v34 = (_DWORD *)*a1;
  v36 = v35;
  v37 = *(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL);
  v38 = *(_WORD *)v37;
  v39 = *(_QWORD *)(v37 + 8);
  LOWORD(v77) = v38;
  v78 = v39;
  if ( v38 )
  {
    v40 = v38 - 17;
    if ( (unsigned __int16)(v38 - 10) > 6u && (unsigned __int16)(v38 - 126) > 0x31u )
    {
      if ( v40 <= 0xD3u )
      {
LABEL_28:
        v41 = v34[17];
        goto LABEL_32;
      }
LABEL_31:
      v41 = v34[15];
      goto LABEL_32;
    }
    if ( v40 <= 0xD3u )
      goto LABEL_28;
  }
  else
  {
    v70 = v36;
    v42 = sub_3007030((__int64)&v77);
    v43 = sub_30070B0((__int64)&v77);
    v36 = v70;
    if ( v43 )
      goto LABEL_28;
    if ( !v42 )
      goto LABEL_31;
  }
  v41 = v34[16];
LABEL_32:
  if ( v41 > 2 )
    BUG();
  v44 = sub_33FAF80(a1[1], 215 - v41, (__int64)&v71, v73, v74, v36, a3);
  if ( v71 )
    sub_B91220((__int64)&v71, v71);
  return v44;
}
