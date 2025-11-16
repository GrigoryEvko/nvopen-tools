// Function: sub_637180
// Address: 0x637180
//
_DWORD *__fastcall sub_637180(
        __int64 a1,
        __int64 *a2,
        __m128i *a3,
        _QWORD *a4,
        unsigned int a5,
        __int64 **a6,
        _QWORD *a7)
{
  __int64 v9; // r12
  __int8 v10; // r13
  bool v11; // bl
  bool v12; // r13
  __int64 v13; // rax
  char v14; // cl
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // r9
  int v18; // r12d
  __int64 **v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v23; // rsi
  __int64 v24; // rcx
  __int64 v25; // rcx
  __int64 v26; // rdx
  __m128i *v27; // r8
  __int64 v28; // rsi
  __int64 v29; // rdi
  char v30; // dl
  __int64 v31; // rax
  _QWORD *v32; // rax
  int v33; // eax
  __int64 v34; // rdi
  int v35; // eax
  __int64 v36; // rsi
  int v37; // eax
  int v38; // eax
  __int64 v39; // rdi
  char j; // al
  __int64 v41; // rsi
  __int64 v42; // [rsp+0h] [rbp-70h]
  __int64 *v43; // [rsp+0h] [rbp-70h]
  unsigned int v44; // [rsp+0h] [rbp-70h]
  unsigned int v45; // [rsp+0h] [rbp-70h]
  unsigned int v46; // [rsp+0h] [rbp-70h]
  unsigned int v47; // [rsp+0h] [rbp-70h]
  unsigned int v49; // [rsp+8h] [rbp-68h]
  int v50; // [rsp+Ch] [rbp-64h]
  bool v52; // [rsp+1Bh] [rbp-55h]
  int i; // [rsp+1Ch] [rbp-54h]
  __int64 *v54; // [rsp+20h] [rbp-50h]
  __int64 *v56; // [rsp+30h] [rbp-40h] BYREF
  __int64 v57[7]; // [rsp+38h] [rbp-38h] BYREF

  v9 = a1;
  v10 = a3[2].m128i_i8[8];
  v11 = (v10 & 0x20) != 0;
  v12 = (v10 & 8) != 0;
  for ( i = dword_4D03C08[0]; *(_BYTE *)(v9 + 140) == 12; v9 = *(_QWORD *)(v9 + 160) )
    ;
  if ( a2 )
  {
    v56 = a2;
    v54 = a2;
    if ( !a4 )
    {
      v50 = 0;
      v15 = 0;
      v52 = 0;
      goto LABEL_12;
    }
    v50 = 0;
  }
  else
  {
    v54 = (__int64 *)sub_6BDB20((a3[2].m128i_i8[10] & 2) != 0, a4);
    if ( !a4 )
    {
      v50 = 1;
      v15 = 0;
      v52 = 0;
      v56 = v54;
      goto LABEL_12;
    }
    if ( (*((_BYTE *)a4 + 179) & 2) != 0 )
      dword_4D03C08[0] = 1;
    v50 = 1;
    v56 = v54;
  }
  v13 = *a4;
  if ( *a4 )
  {
    v14 = *(_BYTE *)(v13 + 80);
    if ( v14 == 9 || v14 == 7 )
    {
      v16 = *(_QWORD *)(v13 + 88);
    }
    else
    {
      v52 = 0;
      v15 = 0;
      if ( v14 != 21 )
        goto LABEL_12;
      v16 = *(_QWORD *)(*(_QWORD *)(v13 + 88) + 192LL);
    }
    v52 = v16 != 0;
    v15 = v16 != 0;
  }
  else
  {
    v52 = 0;
    v15 = 0;
  }
LABEL_12:
  a3[2].m128i_i8[8] &= ~8u;
  if ( dword_4F04C44 != -1
    || (v24 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v24 + 6) & 6) != 0)
    || *(_BYTE *)(v24 + 4) == 12 )
  {
    if ( dword_4F04C64 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 1) != 0 )
    {
      v45 = v15;
      v33 = sub_8D3350(v9);
      v15 = v45;
      if ( v33 )
      {
        v34 = v56[3];
        if ( v34 )
        {
          v35 = sub_6E1B40(v34);
          v15 = v45;
          if ( v35 )
            v9 = *(_QWORD *)&dword_4D03B80;
        }
      }
    }
  }
  switch ( *(_BYTE *)(v9 + 140) )
  {
    case 0:
    case 0xE:
      goto LABEL_30;
    case 5:
      if ( (!dword_4F077BC || qword_4F077A8 <= 0x9EFBu) && !(_DWORD)qword_4F077B4 )
        goto LABEL_16;
      v17 = v56;
      v32 = (_QWORD *)v56[3];
      if ( !v32 )
      {
        if ( *((_BYTE *)v56 + 8) == 1 )
        {
LABEL_52:
          if ( !dword_4D04428 )
          {
            v49 = v15;
            v43 = v17;
            sub_6851C0(29, v17 + 5);
            v15 = v49;
            v17 = v43;
          }
        }
        goto LABEL_17;
      }
      if ( !*v32 )
      {
LABEL_17:
        sub_694AA0(v17, v9, v15, 1, a3);
        v42 = 0;
        v18 = 0;
        goto LABEL_18;
      }
      sub_636FC0((__int64 *)&v56, v9, a3, a3->m128i_i64);
      v18 = 1;
      v42 = 0;
LABEL_18:
      v19 = a6;
      if ( a6 )
      {
LABEL_19:
        *v19 = v54;
        if ( !a2 )
          unk_4F061D8 = *(_QWORD *)sub_6E1A60(v54);
      }
      else
      {
LABEL_31:
        if ( !a2 )
          unk_4F061D8 = *(_QWORD *)sub_6E1A60(v54);
        if ( v50 )
        {
          if ( (a3[2].m128i_i8[11] & 2) != 0 )
            sub_6BD7B0(v54);
          sub_6E1990(v54);
        }
      }
      a3[2].m128i_i8[8] = (8 * v12) | a3[2].m128i_i8[8] & 0xF7;
      if ( !v18 || (a3[2].m128i_i8[9] & 2) != 0 )
      {
        if ( (a3[2].m128i_i8[8] & 8) == 0 )
        {
LABEL_25:
          LOBYTE(v21) = a3[2].m128i_i8[9];
          if ( (v21 & 0xA) != 8 )
            goto LABEL_26;
          goto LABEL_41;
        }
        v20 = a3->m128i_i64[1];
        if ( v20 )
        {
LABEL_24:
          *(_BYTE *)(v20 + 50) |= 0x40u;
          goto LABEL_25;
        }
      }
      else
      {
        v20 = a3->m128i_i64[1];
        if ( v20 )
          goto LABEL_24;
      }
      sub_630880(a3->m128i_i64, v42);
      v20 = a3->m128i_i64[1];
      if ( v20 )
        goto LABEL_24;
      v21 = a3[2].m128i_u8[9];
      if ( (a3[2].m128i_i8[9] & 0xA) != 8 )
        goto LABEL_26;
LABEL_41:
      if ( dword_4F077C4 == 2 )
      {
        if ( (a3[2].m128i_i8[8] & 0x20) != 0 )
        {
          a3[2].m128i_i8[9] = v21 | 2;
        }
        else if ( v52 )
        {
          sub_6854C0(369, a7, *a4);
        }
        else
        {
          sub_6851C0(517, a7);
        }
      }
      else
      {
        sub_685490(370, a7, *a4);
      }
LABEL_26:
      a3[2].m128i_i8[8] = (32 * v11) | a3[2].m128i_i8[8] & 0xDF;
      dword_4D03C08[0] = i;
      return dword_4D03C08;
    case 8:
      if ( v52 && (*((_BYTE *)a4 + 131) & 0x10) != 0 )
      {
        v18 = 1;
        sub_692C90(a4, v56);
        v42 = 0;
        goto LABEL_18;
      }
      v57[0] = v9;
      sub_635980(&v56, v57, a3, a7, a3->m128i_i64);
      v29 = v57[0];
      if ( v57[0] != v9 )
        a4[36] = v57[0];
      v30 = *(_BYTE *)(v29 + 140);
      if ( v30 == 12 )
      {
        v31 = v29;
        do
        {
          v31 = *(_QWORD *)(v31 + 160);
          v30 = *(_BYTE *)(v31 + 140);
        }
        while ( v30 == 12 );
      }
      if ( v30 )
      {
        v39 = sub_8D40F0(v29);
        for ( j = *(_BYTE *)(v39 + 140); j == 12; j = *(_BYTE *)(v39 + 140) )
          v39 = *(_QWORD *)(v39 + 160);
        v42 = 0;
        if ( a5 && (unsigned __int8)(j - 9) <= 2u )
          v42 = sub_630000(v39, a3, (int)a7);
      }
      else
      {
        a3[2].m128i_i8[9] |= 2u;
        v42 = 0;
      }
      goto LABEL_66;
    case 9:
    case 0xA:
    case 0xB:
      if ( (*(_BYTE *)(v9 + 177) & 0x20) != 0
        || (dword_4F04C44 != -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0)
        && (dword_4F077BC || (v46 = v15, v37 = sub_82ED80(v56), v15 = v46, v37)) )
      {
LABEL_30:
        v23 = v9;
        v18 = 1;
        sub_6319F0(v56, v23, a3, a3->m128i_i64);
        v19 = a6;
        v42 = 0;
        if ( a6 )
          goto LABEL_19;
        goto LABEL_31;
      }
      if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v9 + 96LL) + 178LL) & 0x40) == 0 )
      {
        v41 = v9;
        v18 = 0;
        sub_694AA0(v56, v41, v15, a5, a3);
        v42 = 0;
        goto LABEL_18;
      }
      v47 = v15;
      v38 = sub_630EF0((__int64)v56, v9);
      v26 = v47;
      if ( v38 )
      {
        v25 = a5;
        v27 = a3;
        goto LABEL_56;
      }
      v42 = 0;
      if ( a5 )
        v42 = sub_630000(v9, a3, (int)a7);
      sub_6333F0((__int64 *)&v56, v9, a3, (__int64)a7, a3->m128i_i64);
LABEL_66:
      v18 = 1;
      goto LABEL_18;
    case 0xF:
      v44 = v15;
      if ( (unsigned int)sub_630EF0((__int64)v56, v9) )
      {
        v25 = a5;
        v26 = v44;
        v27 = a3;
LABEL_56:
        v28 = v9;
        v18 = 0;
        sub_694AA0(v56, v28, v26, v25, v27);
        v42 = 0;
      }
      else
      {
        v36 = v9;
        v18 = 1;
        sub_636A00((__int64 *)&v56, v36, a3, (__int64)a7, a3->m128i_i64);
        v42 = 0;
      }
      goto LABEL_18;
    default:
LABEL_16:
      v17 = v56;
      if ( *((_BYTE *)v56 + 8) == 1 && !v56[3] )
        goto LABEL_52;
      goto LABEL_17;
  }
}
