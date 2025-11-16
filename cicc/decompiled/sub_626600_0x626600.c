// Function: sub_626600
// Address: 0x626600
//
__int64 __fastcall sub_626600(
        __int64 a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        _DWORD *a8,
        __int64 a9)
{
  int v9; // r15d
  __int64 v10; // rbx
  unsigned __int16 v11; // r8
  bool v12; // dl
  __int64 v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rax
  unsigned int v16; // r15d
  __int16 v17; // r12
  unsigned int v18; // eax
  unsigned int v19; // edx
  unsigned int v20; // r8d
  unsigned __int8 v21; // di
  __int64 v23; // rax
  __int64 v24; // rdi
  __int16 v25; // r8
  __int64 v26; // r9
  int v27; // eax
  __int64 v28; // rax
  int v29; // eax
  __int64 v30; // r8
  __int64 v31; // rcx
  __int64 v32; // rax
  int v33; // eax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  int v37; // eax
  __int64 v38; // rdi
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  char v43; // dl
  __int64 v44; // [rsp+8h] [rbp-88h]
  __int16 v45; // [rsp+10h] [rbp-80h]
  __int64 v46; // [rsp+10h] [rbp-80h]
  __int16 v47; // [rsp+10h] [rbp-80h]
  __int64 v48; // [rsp+10h] [rbp-80h]
  unsigned int v49; // [rsp+18h] [rbp-78h]
  unsigned __int16 v50; // [rsp+18h] [rbp-78h]
  __int64 v51; // [rsp+18h] [rbp-78h]
  int v52; // [rsp+18h] [rbp-78h]
  int v53; // [rsp+18h] [rbp-78h]
  unsigned __int16 v54; // [rsp+18h] [rbp-78h]
  unsigned __int16 v55; // [rsp+18h] [rbp-78h]
  __int16 v56; // [rsp+18h] [rbp-78h]
  __int64 v57; // [rsp+20h] [rbp-70h]
  unsigned int v58; // [rsp+20h] [rbp-70h]
  unsigned int v59; // [rsp+20h] [rbp-70h]
  unsigned int v60; // [rsp+20h] [rbp-70h]
  bool v61; // [rsp+20h] [rbp-70h]
  unsigned int v62; // [rsp+28h] [rbp-68h]
  __int64 v64; // [rsp+38h] [rbp-58h] BYREF
  __int64 v65; // [rsp+40h] [rbp-50h] BYREF
  __int64 v66; // [rsp+48h] [rbp-48h] BYREF
  __int64 v67; // [rsp+50h] [rbp-40h]
  __int64 v68[7]; // [rsp+58h] [rbp-38h] BYREF

  v9 = 1;
  v10 = a2;
  v64 = a1;
  v67 = -1;
  *a8 = 0;
  v62 = 0;
  while ( 1 )
  {
    v11 = word_4F06418[0];
    if ( word_4F06418[0] == 34 )
    {
      *(_WORD *)(v10 + 120) &= 0xC07Fu;
      *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
      if ( !v64 )
      {
        v13 = 6;
        v14 = sub_7259C0(6);
        *(_QWORD *)(v14 + 160) = v64;
LABEL_102:
        v64 = v14;
        goto LABEL_22;
      }
      v35 = sub_8DBED0();
      v24 = v64;
      v25 = 34;
      v26 = v35;
      if ( v35 == v64 )
        goto LABEL_76;
      v59 = 0;
      goto LABEL_47;
    }
    if ( !a3 )
      break;
    if ( word_4F06418[0] == 33 )
    {
      v59 = 0;
      goto LABEL_70;
    }
    v12 = word_4F06418[0] == 52 && unk_4D04474 != 0;
    if ( !v12 )
      break;
    if ( !dword_4F077BC )
      goto LABEL_45;
    if ( dword_4F077C4 != 2 )
      goto LABEL_103;
    if ( unk_4F07778 <= 201102 )
    {
      if ( dword_4F07774 )
      {
LABEL_45:
        *(_WORD *)(v10 + 120) &= 0xC07Fu;
        *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
        if ( !v64 )
        {
          v13 = 6;
          v61 = v12;
          v42 = sub_7259C0(6);
          v43 = v61;
          v14 = v42;
          *(_QWORD *)(v42 + 160) = v64;
LABEL_114:
          *(_BYTE *)(v14 + 168) = *(_BYTE *)(v14 + 168) & 0xFC | (2 * v43 + 1);
          goto LABEL_102;
        }
        v50 = v11;
        v23 = sub_8DBED0();
        v24 = v64;
        v59 = 1;
        v25 = v50;
        v26 = v23;
        if ( v23 == v64 )
          goto LABEL_56;
        goto LABEL_47;
      }
LABEL_103:
      a2 = 2507;
      v55 = word_4F06418[0];
      sub_684B40(&dword_4F063F8, 2507);
      v59 = 1;
      v11 = v55;
      goto LABEL_70;
    }
    v59 = 1;
LABEL_70:
    *(_WORD *)(v10 + 120) &= 0xC07Fu;
    *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
    if ( !v64 )
    {
      v13 = 6;
      v14 = sub_7259C0(6);
      v43 = v59 & 1;
      *(_QWORD *)(v14 + 160) = v64;
      goto LABEL_114;
    }
    v54 = v11;
    v34 = sub_8DBED0();
    v24 = v64;
    v25 = v54;
    v26 = v34;
    if ( v64 == v34 )
      goto LABEL_56;
LABEL_47:
    if ( v26 )
    {
      if ( v24 )
      {
        if ( dword_4F07588 )
        {
          v40 = *(_QWORD *)(v26 + 32);
          if ( *(_QWORD *)(v24 + 32) == v40 )
          {
            if ( v40 )
            {
              if ( v25 == 34 )
                goto LABEL_76;
LABEL_56:
              v52 = 0;
              goto LABEL_57;
            }
          }
        }
      }
    }
    if ( !qword_4D0495C )
    {
      if ( *(_QWORD *)(v24 + 8)
        && (v45 = v25, v51 = v26, v27 = sub_8D2310(v26), v26 = v51, v25 = v45, v27)
        && (v28 = *(_QWORD *)(v51 + 168), !*(_QWORD *)(v28 + 40))
        && (((unsigned __int8)(*(_WORD *)(v28 + 18) >> 7) | *(_BYTE *)(v28 + 18)) & 0x7F) != 0 )
      {
        v48 = v51;
        v56 = v25;
        sub_6851C0(990, dword_4F07508);
        v26 = v48;
        if ( v56 == 34 )
        {
LABEL_76:
          if ( (unsigned int)sub_8D32E0(v26) )
          {
            sub_6851C0(248, dword_4F07508);
LABEL_78:
            v62 = 1;
            v13 = sub_72C930();
          }
          else
          {
            if ( v62 )
              goto LABEL_78;
            v13 = v64;
          }
          a2 = 0;
          v64 = sub_72D2E0(v13, 0);
          goto LABEL_22;
        }
      }
      else if ( v25 == 34 )
      {
        goto LABEL_76;
      }
      goto LABEL_56;
    }
    v47 = v25;
    v44 = v26;
    v37 = sub_623FC0(v24, &v66, &v65, v68);
    v26 = v44;
    v52 = v37;
    if ( v47 == 34 )
    {
      if ( !v37 )
        goto LABEL_76;
      a2 = v65;
      v13 = v66;
      v64 = sub_73F0A0(v66, v65, 0);
      goto LABEL_22;
    }
LABEL_57:
    if ( *(_BYTE *)(v26 + 140) != 12 )
    {
      v46 = v26;
      v29 = sub_8D32E0(v26);
      v26 = v46;
      if ( v29 )
      {
        if ( !v9 )
        {
          a2 = (__int64)dword_4F07508;
          v13 = 249;
          sub_6851C0(249, dword_4F07508);
          v62 = 1;
          v64 = sub_72C930();
          goto LABEL_22;
        }
        v30 = v10 + 72;
        v13 = v64;
        v31 = *(_BYTE *)(v10 + 120) & 0x7F;
        if ( (_BYTE)v31 == 4 )
          v30 = v10 + 80;
        v32 = sub_72D790(v64, v59, 0, v31, v30, 0);
        *(_BYTE *)(v10 + 124) &= ~0x20u;
        v64 = v32;
LABEL_63:
        a2 = v62;
        if ( !v62 )
          goto LABEL_22;
LABEL_64:
        v62 = 1;
        v64 = sub_72C930();
        goto LABEL_22;
      }
    }
    if ( (unsigned int)sub_8D2600(v26) )
    {
      a2 = (__int64)dword_4F07508;
      v13 = 250;
      sub_6851C0(250, dword_4F07508);
      v62 = 1;
      v64 = sub_72C930();
      goto LABEL_22;
    }
    if ( !v52 )
    {
      v13 = v64;
      if ( v59 )
        v36 = sub_72D6A0(v64);
      else
        v36 = sub_72D600(v64);
      v64 = v36;
      goto LABEL_63;
    }
    a2 = v68[0];
    v13 = 473;
    sub_6854E0(473, v68[0]);
    v62 = 1;
    v64 = sub_72C930();
LABEL_22:
    if ( a9 )
      *(_QWORD *)(a9 + 56) = qword_4F063F0;
    unk_4F061D8 = qword_4F063F0;
    sub_7B8B50(v13, a2, qword_4F063F0, v14);
    if ( (unsigned __int16)(word_4F06418[0] - 81) > 0x26u )
    {
      if ( (unsigned __int16)(word_4F06418[0] - 263) > 3u )
        goto LABEL_35;
    }
    else
    {
      v15 = 0x6004000001LL;
      if ( !_bittest64(&v15, (unsigned int)word_4F06418[0] - 81) )
        goto LABEL_35;
    }
    v16 = dword_4F063F8;
    v17 = unk_4F063FC;
    v18 = sub_624060(a9);
    v19 = v18;
    if ( v18 )
    {
      v20 = v18 & 4;
      *(_WORD *)(v10 + 120) = *(_WORD *)(v10 + 120) & 0xC000 | v18 & 0x7F | ((v18 & 3) << 7);
      if ( (v18 & 0xFFFFFFFB) != 0 )
      {
        *(_DWORD *)(v10 + 72) = v16;
        *(_WORD *)(v10 + 76) = v17;
        if ( (v18 & 4) == 0 )
          goto LABEL_29;
      }
      v53 = v18 & 4;
      v60 = v18 & 0xFFFFFFFB;
      v33 = sub_624110(v64, (__int64)dword_4F07508);
      v20 = v53;
      v19 = v60;
      if ( !v33 )
        v20 = 0;
      if ( v60 )
      {
LABEL_29:
        v58 = v20;
        v49 = v19;
        if ( (unsigned int)sub_8D32E0(v64) )
        {
          v38 = 5;
          if ( dword_4D04964 )
            v38 = byte_4F07472[0];
          sub_684AC0(v38, 512);
          v20 = v58;
        }
        else
        {
          v20 = v49 | v58;
        }
      }
      v64 = sub_73C570(v64, v20, v67);
      if ( (*(_BYTE *)(v10 + 124) & 1) != 0 )
      {
        v21 = 7;
        if ( HIDWORD(qword_4F077B4) )
          v21 = (_DWORD)qword_4F077B4 == 0 ? 4 : 7;
        sub_5CC930(v21, 0x44Au);
      }
    }
LABEL_35:
    a2 = (__int64)&v64;
    v9 = 0;
    sub_622ED0(v10, &v64);
  }
  if ( dword_4F077C4 == 2 && (word_4F06418[0] != 1 || (unk_4D04A11 & 2) == 0) )
  {
    a2 = 0;
    v13 = 0;
    if ( !(unsigned int)sub_7C0F00(0, 0) && word_4F06418[0] == 15 )
    {
      *a8 = 1;
      *(_WORD *)(v10 + 120) &= 0xC07Fu;
      *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
      if ( (unk_4D04A12 & 2) == 0 )
      {
        v65 = 0;
        goto LABEL_64;
      }
      v13 = xmmword_4D04A20.m128i_i64[0];
      v65 = xmmword_4D04A20.m128i_i64[0];
      if ( !xmmword_4D04A20.m128i_i64[0] )
        goto LABEL_64;
      if ( (unsigned int)sub_8D2870(xmmword_4D04A20.m128i_i64[0]) )
      {
        a2 = v65;
        v13 = 1764;
        sub_685380(1764, v65);
        v62 = 1;
        v64 = sub_72C930();
        goto LABEL_22;
      }
      if ( v64 )
      {
        v57 = v64;
        if ( (unsigned int)sub_8D2600(v64) )
        {
          sub_6851C0(2335, dword_4F07508);
        }
        else
        {
          if ( !(unsigned int)sub_8D32E0(v57) )
            goto LABEL_17;
          sub_6851C0(2336, dword_4F07508);
        }
        v64 = sub_72C930();
      }
LABEL_17:
      a2 = v65;
      if ( dword_4F077BC )
      {
        if ( unk_4D04A18 )
        {
          if ( (*(_BYTE *)(unk_4D04A18 + 81LL) & 0x40) != 0 )
          {
            v41 = *(_QWORD *)(unk_4D04A18 + 88LL);
            if ( v65 != v41 )
            {
              v65 = *(_QWORD *)(unk_4D04A18 + 88LL);
              a2 = v41;
            }
          }
        }
        v13 = v64;
        if ( v64 )
        {
LABEL_21:
          v64 = sub_73F0A0(v13, a2, 0);
          goto LABEL_22;
        }
      }
      else
      {
        v39 = sub_8D2220(v65);
        v13 = v64;
        v65 = v39;
        a2 = v39;
        if ( v64 )
          goto LABEL_21;
      }
      v13 = a2;
      v64 = sub_72D2B0(a2);
      goto LABEL_22;
    }
  }
  return v64;
}
