// Function: sub_37A0290
// Address: 0x37a0290
//
unsigned __int8 *__fastcall sub_37A0290(__int64 *a1, __int64 a2)
{
  unsigned int v2; // r14d
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v5; // rax
  unsigned __int16 v6; // si
  __int64 v7; // r8
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // r8
  unsigned int v12; // eax
  __int64 v13; // rcx
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rdx
  unsigned __int16 v17; // ax
  unsigned __int64 v18; // r15
  int v19; // eax
  __int128 v20; // rax
  _QWORD *v21; // r12
  __int64 v22; // rax
  __int64 v23; // r9
  __int64 v24; // rsi
  __int64 v25; // r14
  __int64 v26; // rdx
  __int64 v27; // r15
  unsigned int v28; // esi
  unsigned __int8 *v29; // r14
  unsigned __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rdx
  __int128 v35; // [rsp-10h] [rbp-A0h]
  __int64 v36; // [rsp+8h] [rbp-88h]
  __int64 *v37; // [rsp+10h] [rbp-80h]
  unsigned int v38; // [rsp+1Ch] [rbp-74h]
  unsigned int v39; // [rsp+20h] [rbp-70h]
  __int128 v40; // [rsp+20h] [rbp-70h]
  int v41; // [rsp+20h] [rbp-70h]
  unsigned int v42; // [rsp+30h] [rbp-60h] BYREF
  __int64 v43; // [rsp+38h] [rbp-58h]
  __int64 v44; // [rsp+40h] [rbp-50h] BYREF
  __int64 v45; // [rsp+48h] [rbp-48h]
  __int64 v46; // [rsp+50h] [rbp-40h]

  v4 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v8 = a1[1];
  if ( v4 == sub_2D56A50 )
  {
    v9 = v6;
    v10 = *a1;
    sub_2FE6CC0((__int64)&v44, *a1, *(_QWORD *)(v8 + 64), v9, v7);
    LOWORD(v12) = v45;
    LOWORD(v42) = v45;
    v43 = v46;
  }
  else
  {
    v33 = v6;
    v10 = *(_QWORD *)(v8 + 64);
    v12 = v4(*a1, v10, v33, v7);
    v42 = v12;
    v43 = v34;
  }
  if ( (_WORD)v12 )
  {
    if ( (unsigned __int16)(v12 - 176) > 0x34u )
    {
LABEL_5:
      v13 = word_4456340[(unsigned __int16)v42 - 1];
      goto LABEL_8;
    }
  }
  else if ( !sub_3007100((__int64)&v42) )
  {
    goto LABEL_7;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v42 )
  {
    if ( (unsigned __int16)(v42 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_5;
  }
LABEL_7:
  v13 = (unsigned int)sub_3007130((__int64)&v42, v10);
LABEL_8:
  v14 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL);
  v15 = *(unsigned __int16 *)(v14 + 96);
  v16 = *(_QWORD *)(v14 + 104);
  LOWORD(v44) = v15;
  v45 = v16;
  if ( (_WORD)v15 )
  {
    v36 = 0;
    v17 = word_4456580[v15 - 1];
  }
  else
  {
    v41 = v13;
    v17 = sub_3009970((__int64)&v44, v10, v16, v13, v11);
    LODWORD(v13) = v41;
    v36 = v32;
  }
  v38 = v13;
  v18 = 0;
  v37 = *(__int64 **)(a1[1] + 64);
  v39 = v17;
  LOWORD(v19) = sub_2D43050(v17, v13);
  if ( !(_WORD)v19 )
  {
    v19 = sub_3009400(v37, v39, v36, v38, 0);
    HIWORD(v2) = HIWORD(v19);
    v18 = v31;
  }
  LOWORD(v2) = v19;
  *(_QWORD *)&v20 = sub_379AB60((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v21 = (_QWORD *)a1[1];
  v40 = v20;
  v22 = sub_33F7D60(v21, v2, v18);
  v24 = *(_QWORD *)(a2 + 80);
  v25 = v22;
  v27 = v26;
  v44 = v24;
  if ( v24 )
    sub_B96E90((__int64)&v44, v24, 1);
  *((_QWORD *)&v35 + 1) = v27;
  v28 = *(_DWORD *)(a2 + 24);
  *(_QWORD *)&v35 = v25;
  LODWORD(v45) = *(_DWORD *)(a2 + 72);
  v29 = sub_3406EB0(v21, v28, (__int64)&v44, v42, v43, v23, v40, v35);
  if ( v44 )
    sub_B91220((__int64)&v44, v44);
  return v29;
}
