// Function: sub_32FE700
// Address: 0x32fe700
//
__int64 __fastcall sub_32FE700(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        int a10)
{
  int v14; // eax
  char v15; // al
  __int64 v17; // rax
  int v18; // edi
  __int64 v19; // rdx
  __int64 v20; // r9
  __int64 v21; // r8
  __int64 v22; // r15
  __int64 v23; // rsi
  __int64 *v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // r15
  __int64 v27; // [rsp+0h] [rbp-70h]
  __int64 v28; // [rsp+8h] [rbp-68h]
  __int64 *v29; // [rsp+10h] [rbp-60h]
  __int64 v30; // [rsp+10h] [rbp-60h]
  __int64 v31; // [rsp+10h] [rbp-60h]
  __int64 v32; // [rsp+18h] [rbp-58h]
  __int64 v33; // [rsp+20h] [rbp-50h] BYREF
  __int64 v34; // [rsp+28h] [rbp-48h]
  __int64 v35; // [rsp+30h] [rbp-40h] BYREF
  __int64 v36; // [rsp+38h] [rbp-38h]

  v33 = a4;
  v34 = a5;
  v14 = *(_DWORD *)(a8 + 24);
  if ( a10 == 2 )
  {
    if ( v14 != 298 )
      return 0;
    v15 = (*(_BYTE *)(a8 + 33) >> 2) & 3;
    if ( v15 == 2 )
      goto LABEL_8;
    goto LABEL_4;
  }
  if ( v14 != 298 )
    return 0;
  v15 = (*(_BYTE *)(a8 + 33) >> 2) & 3;
  if ( v15 != 3 )
  {
LABEL_4:
    if ( v15 != 1 )
      return 0;
  }
LABEL_8:
  if ( (*(_WORD *)(a8 + 32) & 0x380) != 0 )
    return 0;
  v17 = *(_QWORD *)(a8 + 56);
  if ( !v17 )
    return 0;
  v18 = 1;
  do
  {
    if ( (_DWORD)a9 == *(_DWORD *)(v17 + 8) )
    {
      if ( !v18 )
        return 0;
      v17 = *(_QWORD *)(v17 + 32);
      if ( !v17 )
        goto LABEL_18;
      if ( (_DWORD)a9 == *(_DWORD *)(v17 + 8) )
        return 0;
      v18 = 0;
    }
    v17 = *(_QWORD *)(v17 + 32);
  }
  while ( v17 );
  if ( v18 == 1 )
    return 0;
LABEL_18:
  v19 = *(unsigned __int16 *)(a8 + 96);
  v20 = *(_QWORD *)(a8 + 104);
  v21 = v19;
  if ( a6 || (v22 = *(_QWORD *)(a8 + 112), (*(_BYTE *)(v22 + 37) & 0xF) != 0) || (*(_BYTE *)(a8 + 32) & 8) != 0 )
  {
LABEL_23:
    if ( !(_WORD)v19
      || !(_WORD)v33
      || (((int)*(unsigned __int16 *)(a3 + 2 * (v19 + 274LL * (unsigned __int16)v33 + 71704) + 6) >> (4 * a10)) & 0xF) != 0 )
    {
      return 0;
    }
    v22 = *(_QWORD *)(a8 + 112);
    goto LABEL_27;
  }
  if ( (_WORD)v33 )
  {
    if ( (unsigned __int16)(v33 - 17) > 0xD3u )
      goto LABEL_27;
    goto LABEL_23;
  }
  v31 = *(unsigned __int16 *)(a8 + 96);
  v32 = *(_QWORD *)(a8 + 104);
  if ( sub_30070B0((__int64)&v33) )
    return 0;
  v21 = v31;
  v20 = v32;
LABEL_27:
  v23 = *(_QWORD *)(a8 + 80);
  v24 = *(__int64 **)(a8 + 40);
  v35 = v23;
  if ( v23 )
  {
    v27 = v21;
    v28 = v20;
    v29 = v24;
    sub_B96E90((__int64)&v35, v23, 1);
    v21 = v27;
    v20 = v28;
    v24 = v29;
  }
  LODWORD(v36) = *(_DWORD *)(a8 + 72);
  v26 = sub_33F1B30(a1, a10, (unsigned int)&v35, v33, v34, v22, *v24, v24[1], v24[5], v24[6], v21, v20);
  if ( v35 )
  {
    v30 = v25;
    sub_B91220((__int64)&v35, v35);
    v25 = v30;
  }
  v36 = v25;
  v35 = v26;
  sub_32EB790(a2, a7, &v35, 1, 1);
  sub_34161C0(a1, a8, 1, v26, 1);
  if ( !*(_QWORD *)(a8 + 56) )
    sub_32CF870(a2, a8);
  return a7;
}
