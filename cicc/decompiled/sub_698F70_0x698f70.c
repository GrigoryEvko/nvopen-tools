// Function: sub_698F70
// Address: 0x698f70
//
__int64 __fastcall sub_698F70(__int64 a1, _DWORD *a2, unsigned int a3, __int64 a4)
{
  char v5; // al
  __int64 v6; // r9
  __int64 v7; // r14
  __int64 v8; // r14
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // rax
  char i; // dl
  __int64 v14; // rax
  __int64 v15; // rdi
  char v16; // r8
  __int64 v17; // rax
  char j; // dl
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  char k; // dl
  __int64 v23; // r12
  char v24; // al
  __int64 v25; // r8
  __int64 v26; // rbx
  __int64 v27; // rsi
  __int64 v28; // rdi
  int v29; // ebx
  int v30; // eax
  char v31; // r8
  int v32; // eax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdi
  __int64 v37; // rdi
  __int64 v38; // rdi
  __int64 v39; // [rsp+0h] [rbp-5B0h]
  __int64 v41; // [rsp+18h] [rbp-598h]
  __int64 v42; // [rsp+18h] [rbp-598h]
  char v43; // [rsp+18h] [rbp-598h]
  char v44; // [rsp+23h] [rbp-58Dh]
  int v47; // [rsp+30h] [rbp-580h] BYREF
  unsigned int v48; // [rsp+34h] [rbp-57Ch] BYREF
  __int64 v49; // [rsp+38h] [rbp-578h] BYREF
  _BYTE v50[160]; // [rsp+40h] [rbp-570h] BYREF
  __int64 v51[44]; // [rsp+E0h] [rbp-4D0h] BYREF
  __m128i v52; // [rsp+240h] [rbp-370h] BYREF
  char v53; // [rsp+250h] [rbp-360h]
  __m128i v54[33]; // [rsp+3A0h] [rbp-210h] BYREF

  v5 = *(_BYTE *)(a1 + 72);
  v6 = *(_QWORD *)(a1 + 48);
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  v7 = *(_QWORD *)(a1 + 40);
  v44 = v5;
  v41 = v6;
  v39 = v7;
  sub_6E1E00(4, v50, 0, 0);
  sub_6F8E70(v7, &dword_4F077C8, &dword_4F077C8, v51, 0);
  v8 = v51[0];
  v9 = (__int64)&dword_4F077C8;
  sub_6F8E70(v41, &dword_4F077C8, &dword_4F077C8, &v52, 0);
  LODWORD(v49) = 0;
  v42 = v52.m128i_i64[0];
  if ( (unsigned int)sub_8E3200(v8) || (v10 = v42, (unsigned int)sub_8E3200(v42)) )
  {
    v9 = 0;
    v10 = 31;
    sub_84EC30(
      31,
      0,
      0,
      1,
      0,
      (unsigned int)v51,
      (__int64)&v52,
      (__int64)a2,
      a3,
      1,
      (__int64)a2,
      (__int64)v54,
      0,
      0,
      (__int64)&v49);
  }
  if ( (_DWORD)v49 )
  {
LABEL_6:
    if ( !v54[1].m128i_i8[0] )
      return sub_6E2B30(v10, v9);
    v11 = v54[0].m128i_i64[0];
    for ( i = *(_BYTE *)(v54[0].m128i_i64[0] + 140); i == 12; i = *(_BYTE *)(v11 + 140) )
      v11 = *(_QWORD *)(v11 + 160);
    if ( !i )
      return sub_6E2B30(v10, v9);
    sub_688FA0(v54);
    v14 = sub_6F6F40(v54, 0);
    *(_QWORD *)(a1 + 56) = v14;
    v15 = v14;
    *(_QWORD *)(a1 + 56) = sub_6E2700(v14);
    sub_6E2B30(v15, 0);
    sub_6E1E00(4, v50, 0, 0);
    v9 = (__int64)&dword_4F077C8;
    sub_6F8E70(v39, &dword_4F077C8, &dword_4F077C8, v51, 0);
    v10 = v51[0];
    LODWORD(v49) = 0;
    if ( (unsigned int)sub_8E3200(v51[0]) )
    {
      v9 = 1;
      v10 = 37;
      sub_84EC30(
        37,
        1,
        0,
        1,
        0,
        (unsigned int)v51,
        0,
        (__int64)a2,
        a3,
        2,
        (__int64)a2,
        (__int64)v54,
        0,
        0,
        (__int64)&v49);
    }
    v16 = v44 & 1;
    if ( !(_DWORD)v49 )
    {
      v30 = sub_8D2E30(v51[0]);
      v31 = v44 & 1;
      if ( !v30 )
      {
        v32 = sub_8D2D50(v51[0]);
        v31 = v44 & 1;
        if ( !v32 )
        {
          v9 = (__int64)a2;
          v10 = 2285;
          sub_685360(0x8EDu, a2, v51[0]);
          if ( (v44 & 1) != 0 )
          {
            v9 = (__int64)a2;
            v10 = (__int64)v54;
            sub_6980A0(v54, a2, a3, 0, 0, 0);
          }
          return sub_6E2B30(v10, v9);
        }
      }
      v43 = v31;
      v33 = sub_73D720(v51[0]);
      v9 = 37;
      v10 = (__int64)v51;
      sub_6F7B30(v51, 37, v33, v54);
      v16 = v43;
    }
    if ( v16 )
    {
      v9 = (__int64)a2;
      v10 = (__int64)v54;
      sub_6980A0(v54, a2, a3, 0, 0, 0);
    }
    if ( v54[1].m128i_i8[0] )
    {
      v17 = v54[0].m128i_i64[0];
      for ( j = *(_BYTE *)(v54[0].m128i_i64[0] + 140); j == 12; j = *(_BYTE *)(v17 + 140) )
        v17 = *(_QWORD *)(v17 + 160);
      if ( j )
      {
        v19 = sub_6F6F40(v54, 0);
        *(_QWORD *)(a1 + 64) = v19;
        v20 = v19;
        *(_QWORD *)(a1 + 64) = sub_6E2700(v19);
        sub_6E2B30(v20, 0);
        sub_8601E0(*(_QWORD *)(a1 + 32), a4);
        sub_6E1E00(4, v50, 0, 0);
        sub_6F8E70(*(_QWORD *)(a1 + 40), &dword_4F077C8, &dword_4F077C8, v51, 0);
        v47 = 0;
        if ( !(unsigned int)sub_8E3200(v51[0])
          || (v27 = 1,
              v28 = 7,
              sub_84EC30(
                7,
                1,
                0,
                1,
                0,
                (unsigned int)v51,
                0,
                (__int64)a2,
                a3,
                3,
                (__int64)a2,
                (__int64)&v52,
                0,
                0,
                (__int64)&v47),
              !v47) )
        {
          if ( !(unsigned int)sub_8D2E30(v51[0]) )
          {
            v27 = (__int64)a2;
            v28 = 2286;
            sub_685360(0x8EEu, a2, v51[0]);
            goto LABEL_43;
          }
          sub_6FA3A0(v51);
          v34 = sub_6F6F40(v51, 0);
          v27 = (__int64)&v52;
          v28 = sub_73DCD0(v34);
          sub_6E7150(v28, &v52);
        }
        if ( !v53 )
          goto LABEL_43;
        v21 = v52.m128i_i64[0];
        for ( k = *(_BYTE *)(v52.m128i_i64[0] + 140); k == 12; k = *(_BYTE *)(v21 + 140) )
          v21 = *(_QWORD *)(v21 + 160);
        if ( !k )
          goto LABEL_43;
        v23 = *(_QWORD *)(a1 + 8);
        if ( !v23 )
          goto LABEL_43;
        v24 = *(_BYTE *)(v23 + 175);
        if ( (v24 & 7) != 0 )
        {
          v25 = *(_QWORD *)(v23 + 120);
          v26 = 0;
          if ( (v24 & 4) != 0 )
          {
            v26 = *(_QWORD *)(v23 + 120);
            if ( *(_BYTE *)(v25 + 140) == 12 )
            {
              do
                v26 = *(_QWORD *)(v26 + 160);
              while ( *(_BYTE *)(v26 + 140) == 12 );
            }
          }
          if ( (*(_BYTE *)(v23 + 170) & 2) != 0 )
          {
            if ( !(unsigned int)sub_8D32E0(*(_QWORD *)(v23 + 120)) && (unsigned int)sub_8D3410(v52.m128i_i64[0]) )
            {
              v38 = *(_QWORD *)(v23 + 120);
              v27 = 0;
              if ( (*(_BYTE *)(v38 + 140) & 0xFB) == 8 )
                v27 = (unsigned int)sub_8D4C10(v38, dword_4F077C4 != 2);
              *(_QWORD *)(v23 + 120) = sub_73C570(v52.m128i_i64[0], v27, -1);
              goto LABEL_37;
            }
            v24 = *(_BYTE *)(v23 + 175);
            v25 = *(_QWORD *)(v23 + 120);
          }
          v27 = (v24 & 4) != 0;
          if ( (unsigned int)sub_696CB0(
                               (v24 & 2) != 0,
                               v27,
                               0,
                               0,
                               v25,
                               (_BYTE *)v26,
                               0,
                               (__int64)&v52,
                               0,
                               v23 + 64,
                               &v49,
                               v54[0].m128i_i64,
                               &v48) )
          {
            *(_QWORD *)(v23 + 120) = v49;
          }
          else if ( !v48 )
          {
            v27 = v23 + 64;
            v37 = (*(_BYTE *)(v23 + 175) & 2) == 0 ? 1587 : 2544;
            sub_6851C0(v37, (_DWORD *)(v23 + 64));
            *(_QWORD *)(v23 + 120) = sub_72C930(v37);
          }
        }
LABEL_37:
        if ( (*(_BYTE *)(v23 + 170) & 2) != 0
          && !(unsigned int)sub_8D32E0(*(_QWORD *)(v23 + 120))
          && (v36 = v52.m128i_i64[0], (unsigned int)sub_8D3410(v52.m128i_i64[0])) )
        {
          sub_6E2B30(v36, v27);
          memset(v54, 0, 0x1D8u);
          v54[9].m128i_i64[1] = (__int64)v54;
          v54[1].m128i_i64[1] = *(_QWORD *)&dword_4F063F8;
          if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
            v54[11].m128i_i8[2] |= 1u;
          v54[0].m128i_i64[0] = *(_QWORD *)v23;
          v54[18].m128i_i64[0] = *(_QWORD *)(v23 + 120);
          v54[3].m128i_i64[0] = *(_QWORD *)(v23 + 64);
          v27 = sub_6E3060(&v52);
          sub_692C90((__int64)v54, v27);
          sub_6E1990(v27);
          *(_BYTE *)(v23 + 177) = 2;
          *(_QWORD *)(v23 + 184) = v54[9].m128i_i64[0];
          if ( dword_4D0488C )
          {
            if ( unk_4F04C50 && (*(_BYTE *)(*(_QWORD *)(unk_4F04C50 + 32LL) + 193LL) & 2) != 0 )
              sub_646FB0(v23, v23 + 64);
            return sub_863FA0(0);
          }
          if ( !word_4D04898 )
            return sub_863FA0(0);
          if ( !(_DWORD)qword_4F077B4 )
            return sub_863FA0(0);
          if ( qword_4F077A0 <= 0x765Bu )
            return sub_863FA0(0);
          v28 = dword_4F063F8;
          if ( !(unsigned int)sub_729F80(dword_4F063F8) )
            return sub_863FA0(0);
          v35 = unk_4F04C50;
          if ( !unk_4F04C50 )
            return sub_863FA0(0);
          v29 = 1;
        }
        else
        {
          v27 = (__int64)&v52;
          v28 = v23;
          sub_68BC10(v23, &v52);
          v29 = dword_4D0488C;
          if ( dword_4D0488C )
          {
            if ( unk_4F04C50 )
            {
              if ( (*(_BYTE *)(*(_QWORD *)(unk_4F04C50 + 32LL) + 193LL) & 2) != 0 )
              {
                v27 = v23 + 64;
                v28 = v23;
                sub_646FB0(v23, v23 + 64);
              }
            }
            goto LABEL_43;
          }
          v28 = word_4D04898;
          if ( !word_4D04898
            || !(_DWORD)qword_4F077B4
            || qword_4F077A0 <= 0x765Bu
            || (v28 = dword_4F063F8, !(unsigned int)sub_729F80(dword_4F063F8))
            || (v35 = unk_4F04C50) == 0 )
          {
LABEL_43:
            sub_6E2B30(v28, v27);
            return sub_863FA0(0);
          }
        }
        if ( (*(_BYTE *)(*(_QWORD *)(v35 + 32) + 193LL) & 2) != 0 )
        {
          v27 = v23 + 64;
          v28 = v23;
          sub_646FB0(v23, v23 + 64);
        }
        if ( v29 )
          return sub_863FA0(0);
        goto LABEL_43;
      }
    }
    return sub_6E2B30(v10, v9);
  }
  if ( v8 != v51[0] || v42 != v52.m128i_i64[0] || (unsigned int)sub_8D2EF0(v8) || (unsigned int)sub_8D2870(v8) )
  {
    sub_6FA3A0(v51);
    sub_6FA3A0(&v52);
    v9 = (__int64)&v52;
    v10 = (__int64)v51;
    sub_68F410(v51, v52.m128i_i64, 48, a3, a2, (int)v54);
    goto LABEL_6;
  }
  sub_685360(0x8ECu, a2, v51[0]);
  return sub_6E2B30(2284, a2);
}
