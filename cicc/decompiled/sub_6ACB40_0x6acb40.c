// Function: sub_6ACB40
// Address: 0x6acb40
//
__int64 __fastcall sub_6ACB40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r13d
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // rax
  char j; // dl
  __int64 result; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // rax
  _QWORD *v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r9
  __int64 i; // rax
  __int64 v28; // rax
  __int64 v29; // r10
  char v30; // al
  __int64 v31; // rdi
  __int64 v32; // rsi
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rax
  int v37; // r13d
  __int64 v38; // rbx
  int v39; // eax
  int v40; // eax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // [rsp+0h] [rbp-1B0h]
  __int64 v44; // [rsp+8h] [rbp-1A8h]
  __int64 v45; // [rsp+8h] [rbp-1A8h]
  __int64 v46; // [rsp+8h] [rbp-1A8h]
  __int64 v47; // [rsp+8h] [rbp-1A8h]
  __int64 v48; // [rsp+8h] [rbp-1A8h]
  int v49; // [rsp+14h] [rbp-19Ch] BYREF
  __int64 v50; // [rsp+18h] [rbp-198h] BYREF
  _QWORD v51[2]; // [rsp+20h] [rbp-190h] BYREF
  char v52; // [rsp+30h] [rbp-180h]
  _DWORD v53[5]; // [rsp+64h] [rbp-14Ch] BYREF
  __int64 v54; // [rsp+78h] [rbp-138h]
  __int64 v55; // [rsp+B0h] [rbp-100h]

  v6 = a3;
  v7 = a1;
  v8 = a2;
  v49 = 0;
  if ( a2 )
  {
    v50 = *(_QWORD *)(a2 + 68);
    v9 = qword_4D03C50;
  }
  else
  {
    v50 = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(a1, 0, a3, a4);
    a2 = 125;
    a1 = 27;
    sub_7BE280(27, 125, 0, 0);
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    v9 = qword_4D03C50;
    ++*(_QWORD *)(qword_4D03C50 + 40LL);
  }
  if ( (*(_BYTE *)(v9 + 19) & 0x40) != 0 )
  {
    if ( (unsigned int)sub_6E5430(a1, a2, a3, a4, a5, a6) )
      sub_6851C0(0x39u, &v50);
    goto LABEL_6;
  }
  if ( (unsigned int)sub_6E9250(&v50) )
  {
LABEL_6:
    v49 = 1;
LABEL_7:
    v10 = 0;
    goto LABEL_8;
  }
  if ( v49 )
    goto LABEL_7;
  if ( dword_4F04C58 == -1 )
    goto LABEL_67;
  v24 = qword_4F04C68;
  for ( i = *(_QWORD *)(*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 216) + 152LL);
        *(_BYTE *)(i + 140) == 12;
        i = *(_QWORD *)(i + 160) )
  {
    ;
  }
  if ( (*(_BYTE *)(*(_QWORD *)(i + 168) + 16LL) & 1) == 0 )
    goto LABEL_67;
  v24 = (_QWORD *)unk_4F04C50;
  v28 = *(_QWORD *)(unk_4F04C50 + 40LL);
  if ( v28 )
  {
    do
    {
      v10 = v28;
      v28 = *(_QWORD *)(v28 + 112);
    }
    while ( v28 );
    goto LABEL_8;
  }
  if ( !dword_4F077BC || (v10 = *(_QWORD *)(unk_4F04C50 + 64LL)) == 0 )
  {
LABEL_67:
    if ( (unsigned int)sub_6E5430(&v50, a2, v24, v25, v10, v26) )
      sub_6851C0(0x48Bu, &v50);
    goto LABEL_6;
  }
LABEL_8:
  v44 = v10;
  if ( !v6 )
  {
    ++*(_BYTE *)(qword_4F061C8 + 75LL);
    v11 = sub_6AC060(0, 0x3A0u, &v49);
    ++*(_BYTE *)(qword_4F061C8 + 9LL);
    sub_7BE280(67, 253, 0, 0);
    v12 = qword_4F061C8;
    --*(_BYTE *)(qword_4F061C8 + 9LL);
    --*(_BYTE *)(v12 + 75);
    sub_69ED20((__int64)v51, 0, 0, 1);
    sub_6F69D0(v51, 4);
    v13 = v44;
    if ( v52 != 1 )
    {
      if ( v49 )
      {
LABEL_18:
        sub_6E6260(v7);
        goto LABEL_19;
      }
      if ( v52 != 2 || HIDWORD(qword_4F077B4) == 0 || !v8 )
      {
        if ( !v52 )
          goto LABEL_17;
        v14 = v51[0];
        for ( j = *(_BYTE *)(v51[0] + 140LL); j == 12; j = *(_BYTE *)(v14 + 140) )
          v14 = *(_QWORD *)(v14 + 160);
        if ( !j )
          goto LABEL_17;
        goto LABEL_55;
      }
      v32 = (__int64)v53;
      v31 = 928;
      *(_QWORD *)(v11 + 16) = sub_6F6F40(v51, 0);
      sub_684B30(0x3A0u, v53);
      goto LABEL_45;
    }
    v29 = v55;
    if ( dword_4F077BC )
    {
      if ( qword_4F077A8 <= 0x75F7u || (v41 = sub_6E36E0(v55, 1), v13 = v44, v29 = v41, dword_4F077BC) )
      {
        v30 = *(_BYTE *)(v29 + 24);
        if ( v30 != 1 )
          goto LABEL_39;
        if ( *(_BYTE *)(v29 + 56) != 5 )
          goto LABEL_40;
        v43 = v13;
        v47 = v29;
        v40 = sub_8DED40(*(_QWORD *)v29, **(_QWORD **)(v29 + 72));
        v29 = v47;
        v13 = v43;
        if ( v40 )
          v29 = *(_QWORD *)(v47 + 72);
      }
    }
    v30 = *(_BYTE *)(v29 + 24);
LABEL_39:
    if ( v30 == 3 && ((*(_BYTE *)(v29 + 25) & 1) != 0 || dword_4F077BC) && v13 )
    {
      v42 = *(_QWORD *)(v29 + 56);
      if ( v42 == v13 )
      {
LABEL_43:
        v46 = v29;
        if ( v49 )
          goto LABEL_18;
        v31 = v54;
        v32 = 40;
        sub_6E5820(v54, 40);
        *(_QWORD *)(v11 + 16) = v46;
LABEL_45:
        if ( v49 )
          goto LABEL_18;
        v36 = sub_72CBE0(v31, v32, 0, v33, v34, v35);
        v21 = 111;
        v22 = v36;
        goto LABEL_23;
      }
      if ( *(char *)(v42 + 169) < 0 )
      {
        v48 = v29;
        sub_684B30(0x3A0u, v53);
        v29 = v48;
        goto LABEL_43;
      }
    }
LABEL_40:
    v45 = v29;
    if ( v49 )
      goto LABEL_18;
    if ( !HIDWORD(qword_4F077B4) )
    {
LABEL_55:
      sub_6E68E0(928, v51);
LABEL_17:
      v49 = 1;
      goto LABEL_18;
    }
    sub_684B30(0x3A0u, v53);
    v29 = v45;
    goto LABEL_43;
  }
  v11 = sub_6AC060(0, 0x3A0u, &v49);
  if ( v49 )
    goto LABEL_18;
  v21 = 115;
  v22 = sub_72CBE0(0, 928, v17, v18, v19, v20);
LABEL_23:
  v23 = sub_73DBF0(v21, v22, v11);
  sub_6E70E0(v23, v7);
LABEL_19:
  result = sub_6E26D0(2, v7);
  if ( !v8 )
  {
    v37 = qword_4F063F0;
    v38 = WORD2(qword_4F063F0);
    sub_7BE280(28, 18, 0, 0);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    --*(_QWORD *)(qword_4D03C50 + 40LL);
    v39 = v50;
    *(_DWORD *)(v7 + 76) = v37;
    *(_DWORD *)(v7 + 68) = v39;
    LOWORD(v39) = WORD2(v50);
    *(_WORD *)(v7 + 80) = v38;
    *(_WORD *)(v7 + 72) = v39;
    *(_QWORD *)dword_4F07508 = *(_QWORD *)(v7 + 68);
    unk_4F061D8 = *(_QWORD *)(v7 + 76);
    return sub_6E3280(v7, &v50);
  }
  return result;
}
