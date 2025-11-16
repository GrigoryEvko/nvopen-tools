// Function: sub_33AA110
// Address: 0x33aa110
//
__int64 __fastcall sub_33AA110(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r15
  __int64 v6; // r14
  __int128 v7; // rax
  __int128 v8; // rax
  unsigned __int16 v9; // ax
  unsigned __int8 v10; // r13
  unsigned __int16 v11; // ax
  unsigned __int8 v12; // cl
  int v13; // edx
  __int64 v14; // rax
  __int64 v15; // rsi
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  int v23; // edx
  __int64 v24; // r8
  unsigned __int16 *v25; // rax
  unsigned int v26; // edx
  int v27; // r9d
  __int64 v28; // r15
  int v29; // edx
  int v30; // r14d
  _QWORD *v31; // rax
  __int64 v32; // rsi
  unsigned __int16 v34; // ax
  __int128 v35; // [rsp-20h] [rbp-150h]
  __int128 v36; // [rsp-10h] [rbp-140h]
  int v37; // [rsp+0h] [rbp-130h]
  __int64 v38; // [rsp+8h] [rbp-128h]
  int v39; // [rsp+8h] [rbp-128h]
  int v40; // [rsp+10h] [rbp-120h]
  __int64 v41; // [rsp+10h] [rbp-120h]
  char v42; // [rsp+1Bh] [rbp-115h]
  unsigned int v43; // [rsp+1Ch] [rbp-114h]
  __int128 v44; // [rsp+20h] [rbp-110h]
  __int64 v45; // [rsp+20h] [rbp-110h]
  unsigned __int8 v46; // [rsp+30h] [rbp-100h]
  __int64 v47; // [rsp+38h] [rbp-F8h]
  __int128 v48; // [rsp+40h] [rbp-F0h]
  __int64 v49; // [rsp+90h] [rbp-A0h] BYREF
  int v50; // [rsp+98h] [rbp-98h]
  unsigned __int64 v51; // [rsp+A0h] [rbp-90h]
  __int64 v52; // [rsp+A8h] [rbp-88h]
  __int64 v53; // [rsp+B0h] [rbp-80h]
  unsigned __int64 v54; // [rsp+C0h] [rbp-70h]
  __int64 v55; // [rsp+C8h] [rbp-68h]
  __int64 v56; // [rsp+D0h] [rbp-60h]
  __int64 v57[10]; // [rsp+E0h] [rbp-50h] BYREF

  v3 = sub_338B750(a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v43 = v4;
  v5 = v4;
  v6 = v3;
  v47 = v3;
  *(_QWORD *)&v7 = sub_338B750(a1, *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
  v44 = v7;
  *(_QWORD *)&v8 = sub_338B750(a1, *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
  v48 = v8;
  v9 = sub_33E0440(*(_QWORD *)(a1 + 864), v6, v5);
  if ( HIBYTE(v9) )
  {
    v10 = v9;
    v11 = sub_33E0440(*(_QWORD *)(a1 + 864), v44, *((_QWORD *)&v44 + 1));
    v12 = 0;
    if ( HIBYTE(v11) )
      v12 = v11;
  }
  else
  {
    v34 = sub_33E0440(*(_QWORD *)(a1 + 864), v44, *((_QWORD *)&v44 + 1));
    if ( !HIBYTE(v34) )
    {
      v10 = 0;
      goto LABEL_5;
    }
    v10 = 0;
    v12 = v34;
  }
  if ( v12 >= v10 )
LABEL_5:
    v12 = v10;
  v13 = *(_DWORD *)(a1 + 848);
  v14 = *(_QWORD *)a1;
  v49 = 0;
  v50 = v13;
  if ( v14 )
  {
    if ( &v49 != (__int64 *)(v14 + 48) )
    {
      v15 = *(_QWORD *)(v14 + 48);
      v49 = v15;
      if ( v15 )
      {
        v46 = v12;
        sub_B96E90((__int64)&v49, v15, 1);
        v12 = v46;
      }
    }
  }
  v42 = v12;
  v37 = sub_33738A0(a1);
  v40 = v16;
  v38 = *(_QWORD *)(a1 + 864);
  sub_B91FC0(v57, a2);
  v17 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v18 = *(_QWORD *)(a2 + 32 * (1 - v17));
  if ( v18 )
  {
    v55 = 0;
    BYTE4(v56) = 0;
    v54 = v18 & 0xFFFFFFFFFFFFFFFBLL;
    v19 = *(_QWORD *)(v18 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v19 + 8) - 17 <= 1 )
      v19 = **(_QWORD **)(v19 + 16);
    LODWORD(v18) = *(_DWORD *)(v19 + 8) >> 8;
  }
  else
  {
    v54 = 0;
    v55 = 0;
    BYTE4(v56) = 0;
  }
  LODWORD(v56) = v18;
  v20 = *(_QWORD *)(a2 - 32 * v17);
  if ( v20 )
  {
    BYTE4(v53) = 0;
    v52 = 0;
    v51 = v20 & 0xFFFFFFFFFFFFFFFBLL;
    v21 = *(_QWORD *)(v20 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v21 + 8) - 17 <= 1 )
      v21 = **(_QWORD **)(v21 + 16);
    LODWORD(v20) = *(_DWORD *)(v21 + 8) >> 8;
  }
  else
  {
    v51 = 0;
    v52 = 0;
    BYTE4(v53) = 0;
  }
  LODWORD(v53) = v20;
  v22 = sub_340AD50(
          v38,
          v37,
          v40,
          (unsigned int)&v49,
          v6,
          v5,
          v44,
          v48,
          v42,
          0,
          0,
          0,
          0,
          v51,
          v52,
          v53,
          v54,
          v55,
          v56,
          (__int64)v57,
          0);
  v24 = *(_QWORD *)(a1 + 864);
  if ( v22 )
  {
    v39 = v23;
    v41 = *(_QWORD *)(a1 + 864);
    v45 = v22;
    nullsub_1875(v22, v41, 0);
    *(_QWORD *)(v41 + 384) = v45;
    *(_DWORD *)(v41 + 392) = v39;
    sub_33E2B60(v41, 0);
  }
  else
  {
    *(_QWORD *)(v24 + 384) = 0;
    *(_DWORD *)(v24 + 392) = v23;
  }
  v25 = (unsigned __int16 *)(*(_QWORD *)(v47 + 48) + 16LL * v43);
  *(_QWORD *)&v48 = sub_33FB160(*(_QWORD *)(a1 + 864), v48, *((_QWORD *)&v48 + 1), &v49, *v25, *((_QWORD *)v25 + 1));
  *((_QWORD *)&v36 + 1) = v26 | *((_QWORD *)&v48 + 1) & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v36 = v48;
  *((_QWORD *)&v35 + 1) = v5;
  *(_QWORD *)&v35 = v6;
  v28 = sub_3406EB0(
          *(_QWORD *)(a1 + 864),
          56,
          (unsigned int)&v49,
          *(unsigned __int16 *)(*(_QWORD *)(v47 + 48) + 16LL * v43),
          *(_QWORD *)(*(_QWORD *)(v47 + 48) + 16LL * v43 + 8),
          v27,
          v35,
          v36);
  v30 = v29;
  v57[0] = a2;
  v31 = sub_337DC20(a1 + 8, v57);
  *v31 = v28;
  v32 = v49;
  *((_DWORD *)v31 + 2) = v30;
  if ( v32 )
    sub_B91220((__int64)&v49, v32);
  return 1;
}
