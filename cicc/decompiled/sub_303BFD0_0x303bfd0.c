// Function: sub_303BFD0
// Address: 0x303bfd0
//
__int64 __fastcall sub_303BFD0(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5, int a6)
{
  __int64 v6; // rax
  _QWORD *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // r15
  unsigned __int16 *v13; // rdx
  int v14; // ecx
  __int64 v15; // rdx
  __int64 v16; // rax
  __int16 v17; // ax
  __int64 v18; // rax
  unsigned __int16 v19; // dx
  __int16 v20; // ax
  __int64 v21; // rax
  int v22; // ecx
  int v23; // r8d
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned int v27; // edx
  int v28; // r9d
  __int64 result; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  bool v33; // al
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // rsi
  int v40; // eax
  int v41; // edx
  int v42; // esi
  int v43; // esi
  __int128 v44; // [rsp-10h] [rbp-A0h]
  __int128 v45; // [rsp-10h] [rbp-A0h]
  __int128 v46; // [rsp-10h] [rbp-A0h]
  __int128 v47; // [rsp-10h] [rbp-A0h]
  int v48; // [rsp-8h] [rbp-98h]
  int v49; // [rsp+0h] [rbp-90h]
  __int16 v50; // [rsp+2h] [rbp-8Eh]
  int v51; // [rsp+8h] [rbp-88h]
  __int64 v53; // [rsp+10h] [rbp-80h]
  unsigned __int16 v54; // [rsp+30h] [rbp-60h] BYREF
  __int64 v55; // [rsp+38h] [rbp-58h]
  int v56; // [rsp+40h] [rbp-50h] BYREF
  __int64 v57; // [rsp+48h] [rbp-48h]
  __int64 v58; // [rsp+50h] [rbp-40h] BYREF
  int v59; // [rsp+58h] [rbp-38h]

  v6 = 16LL * (unsigned int)a3;
  v9 = *(_QWORD **)(a2 + 40);
  v10 = *(_QWORD *)(a2 + 48) + v6;
  v11 = *v9;
  v12 = v9[1];
  v13 = (unsigned __int16 *)(*(_QWORD *)(*v9 + 48LL) + 16LL * *((unsigned int *)v9 + 2));
  v14 = *v13;
  v15 = *((_QWORD *)v13 + 1);
  v54 = v14;
  v55 = v15;
  LOWORD(v15) = *(_WORD *)v10;
  v16 = *(_QWORD *)(v10 + 8);
  LOWORD(v56) = v15;
  v57 = v16;
  if ( (_WORD)v14 )
  {
    if ( (unsigned __int16)(v14 - 17) <= 0xD3u )
    {
      if ( word_4456580[v14 - 1] != 10 )
        return a2;
    }
    else if ( (_WORD)v14 != 10 )
    {
      return a2;
    }
  }
  else if ( !sub_30070B0((__int64)&v54) || (unsigned __int16)sub_3009970((__int64)&v54, a2, v30, v31, v32) != 10 )
  {
    return a2;
  }
  v17 = v56;
  if ( (_WORD)v56 )
  {
    if ( (unsigned __int16)(v56 - 17) <= 0xD3u )
      v17 = word_4456580[(unsigned __int16)v56 - 1];
  }
  else
  {
    v33 = sub_30070B0((__int64)&v56);
    v19 = 0;
    if ( !v33 )
      goto LABEL_11;
    v17 = sub_3009970((__int64)&v56, a2, 0, v34, v35);
  }
  if ( v17 == 12 )
  {
    v18 = *(_QWORD *)(a1 + 537016);
    if ( *(_DWORD *)(v18 + 340) <= 0x31Fu || *(_DWORD *)(v18 + 336) <= 0x46u )
    {
      v39 = *(_QWORD *)(a2 + 80);
      v58 = v39;
      if ( v39 )
        sub_B96E90((__int64)&v58, v39, 1);
      *((_QWORD *)&v46 + 1) = v12;
      v59 = *(_DWORD *)(a2 + 72);
      *(_QWORD *)&v46 = v11;
      result = sub_33FAF80(a4, 240, (unsigned int)&v58, v56, v57, a6, v46);
      goto LABEL_40;
    }
  }
  v19 = v56;
LABEL_11:
  v20 = v19;
  if ( !v19 )
  {
    if ( sub_30070B0((__int64)&v56) )
    {
      v20 = sub_3009970((__int64)&v56, a2, v36, v37, v38);
      goto LABEL_14;
    }
    return a2;
  }
  if ( (unsigned __int16)(v19 - 17) <= 0xD3u )
    v20 = word_4456580[v19 - 1];
LABEL_14:
  if ( v20 != 13 )
    return a2;
  v21 = *(_QWORD *)(a1 + 537016);
  if ( *(_DWORD *)(v21 + 340) > 0x383u && *(_DWORD *)(v21 + 336) > 0x4Du )
    return a2;
  if ( !v54 )
  {
    if ( !sub_30070B0((__int64)&v54) )
      goto LABEL_19;
    v40 = sub_3009490(&v54, 0xCu, 0);
    v50 = HIWORD(v40);
    v23 = v41;
LABEL_44:
    HIWORD(v42) = v50;
    LOWORD(v42) = v40;
    v22 = v42;
    goto LABEL_20;
  }
  if ( (unsigned __int16)(v54 - 17) <= 0xD3u )
  {
    v43 = word_4456340[v54 - 1];
    if ( (unsigned __int16)(v54 - 176) > 0x34u )
      LOWORD(v40) = sub_2D43050(12, v43);
    else
      LOWORD(v40) = sub_2D43AD0(12, v43);
    v23 = 0;
    goto LABEL_44;
  }
LABEL_19:
  v22 = 12;
  v23 = 0;
LABEL_20:
  v24 = *(_QWORD *)(a2 + 80);
  v58 = v24;
  if ( v24 )
  {
    v49 = v23;
    v51 = v22;
    sub_B96E90((__int64)&v58, v24, 1);
    v23 = v49;
    v22 = v51;
  }
  v59 = *(_DWORD *)(a2 + 72);
  v25 = *(_QWORD *)(a1 + 537016);
  if ( *(_DWORD *)(v25 + 340) > 0x31Fu && *(_DWORD *)(v25 + 336) > 0x46u )
  {
    *((_QWORD *)&v47 + 1) = v12;
    *(_QWORD *)&v47 = v11;
    v26 = sub_33FAF80(a4, 233, (unsigned int)&v58, v22, v23, a6, v47);
    v28 = v48;
  }
  else
  {
    *((_QWORD *)&v44 + 1) = v12;
    *(_QWORD *)&v44 = v11;
    v26 = sub_33FAF80(a4, 240, (unsigned int)&v58, v22, v23, a6, v44);
  }
  *((_QWORD *)&v45 + 1) = v27 | a3 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v45 = v26;
  result = sub_33FAF80(a4, 233, (unsigned int)&v58, v56, v57, v28, v45);
LABEL_40:
  if ( v58 )
  {
    v53 = result;
    sub_B91220((__int64)&v58, v58);
    return v53;
  }
  return result;
}
