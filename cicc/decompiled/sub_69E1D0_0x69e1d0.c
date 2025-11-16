// Function: sub_69E1D0
// Address: 0x69e1d0
//
__int64 __fastcall sub_69E1D0(__int64 *a1, __int64 a2)
{
  unsigned int v2; // r14d
  __int64 v3; // rdi
  unsigned int i; // eax
  unsigned int v5; // eax
  unsigned int *v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rax
  unsigned __int64 v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 result; // rax
  char v15; // si
  __int64 v16; // rdx
  __int64 v17; // rcx
  _QWORD *v18; // r15
  __int64 v19; // rax
  bool v20; // zf
  __int64 v21; // rsi
  unsigned __int16 v22; // ax
  __int64 v23; // rsi
  bool v24; // cc
  __int32 v25; // ebx
  __int64 v26; // rax
  char v27; // dl
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rbx
  int v38; // eax
  int v39; // ebx
  __int64 v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rdi
  __int64 v43; // r12
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // rax
  __int64 v57; // rcx
  __int64 v58; // rdx
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // rax
  __int64 v66; // rdi
  __int64 v67; // rdx
  __int64 v68; // rcx
  unsigned __int16 v69; // ax
  __int64 v70; // rax
  __int64 v71; // [rsp+8h] [rbp-488h]
  __int64 v72; // [rsp+18h] [rbp-478h]
  bool v73; // [rsp+35h] [rbp-45Bh]
  unsigned __int16 v74; // [rsp+36h] [rbp-45Ah]
  __int64 *v75; // [rsp+38h] [rbp-458h]
  unsigned int v76; // [rsp+40h] [rbp-450h]
  unsigned int v77; // [rsp+44h] [rbp-44Ch]
  int v79; // [rsp+5Ch] [rbp-434h] BYREF
  __int64 v80; // [rsp+60h] [rbp-430h] BYREF
  __int64 v81; // [rsp+68h] [rbp-428h] BYREF
  __m128i v82; // [rsp+70h] [rbp-420h] BYREF
  _BYTE v83[160]; // [rsp+80h] [rbp-410h] BYREF
  _QWORD v84[44]; // [rsp+120h] [rbp-370h] BYREF
  _QWORD v85[66]; // [rsp+280h] [rbp-210h] BYREF

  v2 = dword_4F06650[0];
  v77 = unk_4F07270;
  v76 = dword_4F063F8;
  v74 = word_4F063FC[0];
  v3 = *qword_4D03C00;
  for ( i = dword_4F06650[0]; ; i = v5 + 1 )
  {
    v5 = qword_4D03C00[1] & i;
    v6 = (unsigned int *)(v3 + 24LL * v5);
    v7 = *v6;
    if ( dword_4F06650[0] == (_DWORD)v7 )
    {
      a2 = *((_QWORD *)v6 + 1);
      v7 = v6[2];
      v8 = *((_QWORD *)v6 + 2);
      goto LABEL_6;
    }
    if ( !(_DWORD)v7 )
      break;
  }
  v8 = 0;
LABEL_6:
  v82.m128i_i64[1] = v8;
  v9 = a2 & 0xFFFFFFFF00000000LL;
  v82.m128i_i64[0] = v9 | v7;
  sub_7B8B50(v3, v9, v9 | v7, 0xFFFFFFFF00000000LL);
  v11 = v82.m128i_u32[0];
  if ( v82.m128i_i32[0] )
  {
    do
      sub_7B8B50(v3, v9, v11, v10);
    while ( v82.m128i_i32[0] >= dword_4F06650[0] && word_4F06418[0] != 9 );
    if ( dword_4F04C44 == -1
      && (v12 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v12 + 6) & 2) == 0)
      && (unk_4F04C48 == -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 6) & 0x10) == 0)
      && ((*(_BYTE *)(v12 + 6) & 4) == 0 || (*(_BYTE *)(v12 + 12) & 0x10) != 0)
      && *(_BYTE *)(v82.m128i_i64[1] + 24) )
    {
      sub_89F7D0(v85);
      v37 = unk_4F04C18;
      sub_865900(0);
      if ( v37 )
      {
        unk_4F04C18 = v37;
        *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) &= ~4u;
        v38 = sub_6F2300(v82.m128i_i64[1], v85);
        unk_4F04C18 = 0;
        v39 = v38;
      }
      else
      {
        *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) &= ~4u;
        v39 = sub_6F2300(v82.m128i_i64[1], v85);
      }
      sub_864110();
      sub_6E7080(a1, v39);
      v40 = sub_72C390();
      v41 = v85[0];
      *a1 = v40;
      a1[34] = v40;
      return sub_823A00(v41, 24LL * v85[1]);
    }
    else
    {
      v13 = sub_73B8B0(v82.m128i_i64[1], 0x4000);
      return sub_6E70E0(v13, a1);
    }
  }
  v15 = *(_BYTE *)(qword_4D03C50 + 19LL);
  v73 = (v15 & 2) != 0;
  *(_BYTE *)(qword_4D03C50 + 19LL) = v15 | 2;
  ++*(_BYTE *)(qword_4F061C8 + 82LL);
  memset(v85, 0, 0x1D8u);
  v85[19] = v85;
  v85[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v85[22]) |= 1u;
  BYTE5(v85[16]) |= 1u;
  if ( word_4F06418[0] == 27 )
  {
    v18 = (_QWORD *)sub_62B780((__int64)v85);
    v19 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( dword_4F04C44 != -1 )
      goto LABEL_21;
  }
  else
  {
    sub_8600D0(1, 0xFFFFFFFFLL, 0, 0);
    v18 = 0;
    v19 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    *(_BYTE *)(v19 + 11) |= 0x40u;
    v20 = dword_4F04C44 == -1;
    *(_QWORD *)(v19 + 624) = v85;
    if ( !v20 )
    {
LABEL_21:
      *(_BYTE *)(v19 + 6) |= 0x80u;
      goto LABEL_22;
    }
  }
  if ( (*(_BYTE *)(v19 + 6) & 6) != 0 || *(_BYTE *)(v19 + 4) == 12 )
    goto LABEL_21;
LABEL_22:
  if ( unk_4F073B8 != v77 )
  {
    sub_7296B0(v77, &dword_4F04C44, v16, v17);
    v34 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    v35 = *(_QWORD *)(v34 + 496);
    *(_DWORD *)(v34 + 192) = v77;
    *(_QWORD *)(v34 + 488) = v35;
    qword_4F06BC0 = v35;
    if ( word_4F06418[0] != 73 )
      goto LABEL_24;
LABEL_49:
    v81 = *(_QWORD *)&dword_4F063F8;
    goto LABEL_25;
  }
  if ( word_4F06418[0] == 73 )
    goto LABEL_49;
LABEL_24:
  v81 = *(_QWORD *)&dword_4F077C8;
LABEL_25:
  v21 = 130;
  if ( !(unsigned int)sub_7BE280(73, 130, 0, 0) )
  {
    v30 = (__int64)a1;
    sub_6E6260(a1);
    goto LABEL_38;
  }
  v72 = sub_726700(33);
  *(_QWORD *)(v72 + 64) = v18;
  v75 = (__int64 *)(v72 + 56);
  v22 = word_4F06418[0];
  if ( word_4F06418[0] == 74 )
  {
    sub_6E5C80(7, 3112, &dword_4F063F8);
    v22 = word_4F06418[0];
  }
  v23 = (__int64)&v80;
  v24 = v22 <= 0xB7u;
  if ( v22 == 183 )
    goto LABEL_67;
  while ( 1 )
  {
    if ( !v24 )
    {
      if ( v22 == 294 )
      {
        v43 = sub_726700(35);
        *(_QWORD *)v43 = sub_72CBE0(35, v23, v44, v45, v46, v47);
        *(_QWORD *)(v43 + 28) = *(_QWORD *)&dword_4F063F8;
        sub_7B8B50(35, v23, v48, v49);
        ++*(_BYTE *)(qword_4F061C8 + 83LL);
        *(_QWORD *)(v43 + 56) = sub_6D6A30();
        goto LABEL_64;
      }
LABEL_59:
      if ( !(unsigned int)sub_692B20(v22) )
        goto LABEL_32;
      ++*(_BYTE *)(qword_4F061C8 + 83LL);
      sub_6E1DD0(&v80);
      sub_6E1E00(5, v83, 0, 1);
      sub_69ED20(v84, 0, 0, 0);
      sub_6F6C80(v84);
      v42 = sub_6F6F40(v84, 0);
      v43 = sub_6E2700(v42);
      sub_6E2B30(v42, 0);
      sub_6E1DF0(v80);
      unk_4F061D8 = *(_QWORD *)&dword_4F063F8;
      v23 = 65;
      sub_7BE280(75, 65, 0, 0);
      --*(_BYTE *)(qword_4F061C8 + 83LL);
      *v75 = v43;
      if ( !v43 )
        goto LABEL_66;
      goto LABEL_65;
    }
    if ( v22 != 73 )
      break;
    v43 = sub_726700(34);
    *(_QWORD *)v43 = sub_72CBE0(34, v23, v59, v60, v61, v62);
    *(_QWORD *)(v43 + 28) = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(34, v23, v63, v64);
    v65 = qword_4F061C8;
    ++*(_BYTE *)(qword_4F061C8 + 83LL);
    ++*(_BYTE *)(v65 + 82);
    sub_6E1DD0(&v80);
    sub_6E1E00(5, v83, 0, 1);
    sub_69ED20(v84, 0, 0, 1);
    sub_6F6C80(v84);
    v66 = sub_6F6F40(v84, 0);
    v71 = sub_6E2700(v66);
    sub_6E2B30(v66, 0);
    sub_6E1DF0(v80);
    unk_4F061D8 = *(_QWORD *)&dword_4F063F8;
    sub_7BE280(74, 67, 3196, v43 + 28);
    --*(_BYTE *)(qword_4F061C8 + 82LL);
    *(_QWORD *)(v43 + 56) = v71;
    v69 = word_4F06418[0];
    if ( word_4F06418[0] == 243 )
    {
      *(_BYTE *)(v43 + 64) |= 1u;
      sub_7B8B50(74, &qword_4F061C8, v67, v68);
      v69 = word_4F06418[0];
    }
    if ( v69 == 30 )
    {
      v79 = 0;
      sub_7B8B50(74, &qword_4F061C8, v67, v68);
      if ( dword_4F077C4 == 2 )
      {
        if ( (word_4F06418[0] != 1 || (unk_4D04A11 & 2) == 0) && !(unsigned int)sub_7C0F00(0x4000, 0) )
        {
LABEL_76:
          sub_6851D0(0xBF1u);
          goto LABEL_64;
        }
      }
      else if ( word_4F06418[0] != 1 )
      {
        goto LABEL_76;
      }
      v70 = sub_7BF130(16385, 11, &v79);
      if ( !v70 || *(_BYTE *)(v70 + 80) != 22 )
        goto LABEL_76;
      *(_QWORD *)(v71 + 16) = sub_8988D0(v70, 1);
    }
LABEL_64:
    while ( 2 )
    {
      v23 = 65;
      sub_7BE280(75, 65, 0, 0);
      --*(_BYTE *)(qword_4F061C8 + 83LL);
      *v75 = v43;
LABEL_65:
      v75 = (__int64 *)(v43 + 16);
LABEL_66:
      v22 = word_4F06418[0];
      v24 = word_4F06418[0] <= 0xB7u;
      if ( word_4F06418[0] == 183 )
      {
LABEL_67:
        v43 = sub_726700(22);
        *(_QWORD *)v43 = sub_72CBE0(22, v23, v50, v51, v52, v53);
        *(_QWORD *)(v43 + 28) = *(_QWORD *)&dword_4F063F8;
        sub_7B8B50(22, v23, v54, v55);
        v56 = sub_6512E0(0, 0, 0, 1, 0, 0);
        v58 = qword_4F061C8;
        ++*(_BYTE *)(qword_4F061C8 + 83LL);
        if ( v56 )
        {
          *(_QWORD *)(v43 + 56) = *(_QWORD *)(v56 + 88);
          sub_7B8B50(0, &qword_4F061C8, v58, v57);
        }
        else
        {
          sub_6851D0(0xBEEu);
          sub_7264E0(v43, 0);
        }
        continue;
      }
      break;
    }
  }
  if ( v22 != 74 && v22 != 9 )
    goto LABEL_59;
LABEL_32:
  *(_QWORD *)v72 = sub_72C390();
  v25 = dword_4F06650[0];
  *(_DWORD *)(v72 + 28) = v76;
  *(_WORD *)(v72 + 32) = v74;
  unk_4F061D8 = qword_4F063F0;
  sub_7BE280(74, 67, 3196, &v81);
  if ( dword_4F04C44 != -1
    || (v26 = qword_4F04C68[0] + 776LL * dword_4F04C64, v27 = *(_BYTE *)(v26 + 6), (v27 & 6) != 0)
    || *(_BYTE *)(v26 + 4) == 12
    || v27 < 0 )
  {
    v30 = v72;
    sub_6E70E0(v72, a1);
  }
  else
  {
    sub_89F7D0(v84);
    v28 = (int)sub_6F2300(v72, v84);
    sub_6E7080(a1, v28);
    v29 = sub_72C390();
    a1[36] = v72;
    v30 = v84[0];
    *a1 = v29;
    a1[34] = v29;
    *(__int64 *)((char *)a1 + 68) = *(_QWORD *)(v72 + 28);
    sub_823A00(v30, 24LL * v84[1]);
  }
  if ( dword_4F04C44 != -1
    || (v21 = (__int64)qword_4F04C68, v36 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v36 + 6) & 6) != 0)
    || *(_BYTE *)(v36 + 4) == 12 )
  {
    v21 = v2;
    v82.m128i_i32[0] = v25;
    v82.m128i_i64[1] = v72;
    v30 = (__int64)qword_4D03C00;
    sub_69DF50(qword_4D03C00, v2, &v82, v2);
  }
LABEL_38:
  if ( unk_4F073B8 != v77 )
  {
    for ( ; v18; v18 = (_QWORD *)*v18 )
    {
      while ( *(v18 - 2) )
      {
        v18 = (_QWORD *)*v18;
        if ( !v18 )
          goto LABEL_44;
      }
      v21 = 3;
      sub_729FB0(v18, 3, unk_4D03FF0);
    }
LABEL_44:
    v30 = v77;
    sub_7296B0(v77, v21, v31, v32);
  }
  sub_863FC0(v30, v21, v31, v32, v33);
  --*(_BYTE *)(qword_4F061C8 + 82LL);
  result = *(_BYTE *)(qword_4D03C50 + 19LL) & 0xFD | (2 * (unsigned int)v73);
  *(_BYTE *)(qword_4D03C50 + 19LL) = *(_BYTE *)(qword_4D03C50 + 19LL) & 0xFD | (2 * v73);
  return result;
}
