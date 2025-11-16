// Function: sub_6AA570
// Address: 0x6aa570
//
__int64 __fastcall sub_6AA570(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // ebx
  __int16 v5; // r13
  __int64 v6; // r14
  char v7; // dl
  __int64 v8; // rax
  __int64 v10; // rax
  char i; // dl
  __int64 v12; // rax
  char j; // dl
  __int64 v14; // rax
  __int64 v15; // rdi
  char v16; // al
  __int64 v17; // r14
  __int64 v18; // r8
  char v19; // al
  __int64 v20; // rax
  __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  __int64 v28; // rdx
  bool v29; // zf
  __int64 v30; // rax
  int v31; // eax
  char v32; // al
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // [rsp+18h] [rbp-488h]
  __int64 v42; // [rsp+28h] [rbp-478h]
  char v43; // [rsp+34h] [rbp-46Ch] BYREF
  __int64 v44; // [rsp+38h] [rbp-468h] BYREF
  __int64 v45; // [rsp+40h] [rbp-460h] BYREF
  __int64 v46; // [rsp+48h] [rbp-458h] BYREF
  char v47; // [rsp+50h] [rbp-450h] BYREF
  __int64 v48; // [rsp+94h] [rbp-40Ch]
  __int64 v49; // [rsp+E0h] [rbp-3C0h]
  _QWORD v50[2]; // [rsp+1B0h] [rbp-2F0h] BYREF
  char v51; // [rsp+1C0h] [rbp-2E0h]
  _DWORD v52[71]; // [rsp+1F4h] [rbp-2ACh] BYREF
  _BYTE v53[68]; // [rsp+310h] [rbp-190h] BYREF
  __int64 v54; // [rsp+354h] [rbp-14Ch]
  __int64 v55; // [rsp+35Ch] [rbp-144h]

  if ( a1 )
  {
    sub_6F8AB0(a1, (unsigned int)&v47, (unsigned int)v50, 0, (unsigned int)&v45, (unsigned int)&v43, 0);
    v6 = *(_QWORD *)(v49 + 56);
    v44 = v6;
    v46 = v48;
  }
  else
  {
    v45 = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(0, a2, a3, a4);
    sub_7BE280(27, 125, 0, 0);
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    v46 = *(_QWORD *)&dword_4F063F8;
    sub_65CD60(&v44);
    sub_7BE280(67, 253, 0, 0);
    sub_69ED20((__int64)v50, 0, 0, 1);
    v4 = qword_4F063F0;
    v5 = WORD2(qword_4F063F0);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    sub_7BE280(28, 18, 0, 0);
    v6 = v44;
  }
  v7 = *(_BYTE *)(v6 + 140);
  if ( v7 == 12 )
  {
    v8 = v6;
    do
    {
      v8 = *(_QWORD *)(v8 + 160);
      v7 = *(_BYTE *)(v8 + 140);
    }
    while ( v7 == 12 );
  }
  if ( !v7 || !v51 )
    return sub_6E6260(a2);
  v10 = v50[0];
  for ( i = *(_BYTE *)(v50[0] + 140LL); i == 12; i = *(_BYTE *)(v10 + 140) )
    v10 = *(_QWORD *)(v10 + 160);
  if ( !i )
    return sub_6E6260(a2);
  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v6) )
    sub_8AE000(v6);
  sub_6F69D0(v50, 2);
  if ( !v51 )
    return sub_6E6260(a2);
  v12 = v50[0];
  for ( j = *(_BYTE *)(v50[0] + 140LL); j == 12; j = *(_BYTE *)(v12 + 140) )
    v12 = *(_QWORD *)(v12 + 160);
  if ( !j )
    return sub_6E6260(a2);
  if ( dword_4F04C44 != -1
    || (v14 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v14 + 6) & 6) != 0)
    || *(_BYTE *)(v14 + 4) == 12 )
  {
    if ( (unsigned int)sub_8DBE70(v44) || (unsigned int)sub_8DBE70(v50[0]) )
      goto LABEL_32;
  }
  v15 = v44;
  v16 = *(_BYTE *)(v44 + 140);
  if ( v16 == 12 )
  {
    v17 = sub_8D4A00(v44);
LABEL_25:
    v18 = v50[0];
    v19 = *(_BYTE *)(v50[0] + 140LL);
    if ( v19 != 12 )
    {
      v15 = v44;
      if ( dword_4F077C0 && (v19 == 1 || v19 == 7) )
      {
        v20 = 1;
        goto LABEL_28;
      }
LABEL_27:
      v20 = *(_QWORD *)(v18 + 128);
      goto LABEL_28;
    }
LABEL_54:
    v20 = sub_8D4A00(v18);
    v15 = v44;
LABEL_28:
    if ( v20 != v17 )
    {
      sub_6E5ED0(3085, v52, v15, v50[0]);
      return sub_6E6260(a2);
    }
    goto LABEL_46;
  }
  if ( !dword_4F077C0 || v16 != 1 && v16 != 7 )
  {
    v17 = *(_QWORD *)(v44 + 128);
    goto LABEL_25;
  }
  v18 = v50[0];
  v32 = *(_BYTE *)(v50[0] + 140LL);
  if ( v32 == 12 )
  {
    v17 = 1;
    goto LABEL_54;
  }
  if ( v32 != 1 )
  {
    v17 = 1;
    if ( v32 != 7 )
      goto LABEL_27;
  }
LABEL_46:
  if ( !(unsigned int)sub_8E3AD0(v15) )
  {
    if ( (unsigned int)sub_6E5430(v15, 2, v33, v34, v35, v36) )
      sub_6851C0(0xC0Eu, &v46);
    return sub_6E6260(a2);
  }
  if ( !(unsigned int)sub_8E3AD0(v50[0]) )
  {
    sub_69A8C0(3086, v52, v37, v38, v39, v40);
    return sub_6E6260(a2);
  }
LABEL_32:
  v21 = sub_726700(23);
  v22 = v44;
  *(_BYTE *)(v21 + 56) = 71;
  *(_QWORD *)v21 = v22;
  v42 = sub_726700(22);
  v27 = sub_72CBE0(22, 2, v23, v24, v25, v26);
  v28 = v42;
  *(_QWORD *)v42 = v27;
  v29 = *(_BYTE *)(v42 + 24) == 22;
  *(_QWORD *)(v42 + 56) = v44;
  *(_QWORD *)(v42 + 28) = v46;
  if ( v29 && (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 )
  {
    sub_6E70E0(v42, v53);
    LODWORD(v54) = v45;
    WORD2(v54) = WORD2(v45);
    *(_QWORD *)dword_4F07508 = v54;
    v55 = *(_QWORD *)&dword_4F077C8;
    unk_4F061D8 = *(_QWORD *)&dword_4F077C8;
    sub_6E3280(v53, &dword_4F077C8);
    sub_6F6F40(v53, 0);
    v28 = v42;
  }
  v41 = v28;
  v30 = sub_6F6F40(v50, 0);
  *(_QWORD *)(v41 + 16) = v30;
  *(_BYTE *)(v30 + 25) &= ~1u;
  *(_QWORD *)(v21 + 64) = v41;
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 && *(_BYTE *)(v21 + 24) )
  {
    sub_6E70E0(v21, v53);
    LODWORD(v55) = v4;
    LODWORD(v54) = v45;
    WORD2(v55) = v5;
    WORD2(v54) = WORD2(v45);
    *(_QWORD *)dword_4F07508 = v54;
    unk_4F061D8 = v55;
    sub_6E3280(v53, &dword_4F077C8);
    sub_6F6F40(v53, 0);
  }
  sub_6E70E0(v21, a2);
  v31 = v45;
  *(_DWORD *)(a2 + 76) = v4;
  *(_WORD *)(a2 + 80) = v5;
  *(_DWORD *)(a2 + 68) = v31;
  *(_WORD *)(a2 + 72) = WORD2(v45);
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a2 + 68);
  unk_4F061D8 = *(_QWORD *)(a2 + 76);
  return sub_6E3280(a2, &v45);
}
