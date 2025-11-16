// Function: sub_6C4500
// Address: 0x6c4500
//
__int64 __fastcall sub_6C4500(char a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 i; // r15
  __int64 v6; // rax
  char v7; // r12
  bool v8; // r12
  __int64 v9; // rsi
  char v10; // al
  _BYTE *v11; // rdi
  unsigned int v12; // r15d
  __int64 v14; // r14
  __int64 v15; // r15
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // rax
  __int64 k; // rdx
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r15
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rax
  char j; // dl
  __int64 v31; // rax
  char v32; // al
  __int64 v33; // rdi
  char v34; // al
  __int64 v35; // rax
  char v36; // al
  __int64 v37; // rax
  char v38; // dl
  __int64 v39; // [rsp+8h] [rbp-3F8h]
  __int64 v40; // [rsp+10h] [rbp-3F0h]
  __int64 v41; // [rsp+10h] [rbp-3F0h]
  __int64 v42; // [rsp+18h] [rbp-3E8h]
  __int64 v43; // [rsp+18h] [rbp-3E8h]
  unsigned int v44; // [rsp+20h] [rbp-3E0h]
  int v46; // [rsp+34h] [rbp-3CCh] BYREF
  __int64 v47; // [rsp+38h] [rbp-3C8h] BYREF
  _BYTE v48[16]; // [rsp+40h] [rbp-3C0h] BYREF
  _BYTE v49[8]; // [rsp+50h] [rbp-3B0h] BYREF
  __int64 v50; // [rsp+58h] [rbp-3A8h]
  _QWORD *v51; // [rsp+60h] [rbp-3A0h]
  _BYTE v52[160]; // [rsp+70h] [rbp-390h] BYREF
  __m128i v53[22]; // [rsp+110h] [rbp-2F0h] BYREF
  __m128i v54; // [rsp+270h] [rbp-190h] BYREF
  unsigned __int8 v55; // [rsp+280h] [rbp-180h]
  __int64 v56; // [rsp+300h] [rbp-100h]

  v4 = *(_QWORD *)(*(_QWORD *)(a3 + 64) + 16LL);
  for ( i = sub_73D6E0(a2); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  sub_6E1DD0(&v47);
  sub_6E1E00(5, v52, 0, 1);
  *(_DWORD *)(qword_4D03C50 + 18LL) |= 0x10080u;
  v6 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v7 = *(_BYTE *)(v6 + 7);
  *(_BYTE *)(v6 + 7) = v7 & 0xF7;
  v8 = (v7 & 8) != 0;
  sub_6E1BE0(v48);
  v9 = (__int64)v53;
  sub_6E2E50(0, v53);
  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(i) )
    sub_8AE000(i);
  v10 = *(_BYTE *)(i + 140);
  switch ( v10 )
  {
    case 13:
      v42 = sub_8D4890(i);
      v22 = sub_8D4870(i);
      v44 = sub_8D2310(v22);
      if ( !v4 )
        goto LABEL_8;
      v9 = v44;
      if ( !v44 )
      {
        if ( *(_QWORD *)(v4 + 16) )
          goto LABEL_8;
      }
      v40 = *(_QWORD *)(v4 + 56);
      v39 = sub_73D6E0(v40);
      v23 = v39;
      if ( (unsigned int)sub_8D3A70(v39) )
      {
        v9 = v42;
        if ( (unsigned int)sub_8D5DF0(v39) )
        {
LABEL_45:
          v43 = sub_68B9A0(v40);
          if ( !v43 )
            goto LABEL_71;
          v27 = sub_68B9A0(a2);
          if ( !v27 )
          {
            v12 = 0;
            sub_6E1990(v43);
            goto LABEL_72;
          }
          sub_6E1C20(v27, 0, v48);
          *(_QWORD *)(qword_4D03C50 + 136LL) = v48;
          sub_7ADF70(v49, 0);
          dword_4F06648 += 2;
          v28 = sub_7AE2C0(147, dword_4F06648, &dword_4F063F8);
          if ( v50 )
            *v51 = v28;
          else
            v50 = v28;
          v51 = (_QWORD *)v28;
          sub_7BC000(v49);
          v9 = 0;
          sub_6B0A80(*(_QWORD *)(v43 + 24) + 8LL, 0, 0, (__int64)&v54, v53, v43);
          sub_6E1990(v43);
          if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) != 0 || !v55 )
            goto LABEL_71;
          v29 = v54.m128i_i64[0];
          for ( j = *(_BYTE *)(v54.m128i_i64[0] + 140); j == 12; j = *(_BYTE *)(v29 + 140) )
            v29 = *(_QWORD *)(v29 + 160);
          if ( j )
          {
            v12 = 1;
            if ( v44 )
            {
              v31 = sub_6E3060(&v54);
              v4 = *(_QWORD *)(v4 + 16);
              v14 = v31;
              goto LABEL_11;
            }
          }
          else
          {
LABEL_71:
            v12 = 0;
          }
LABEL_72:
          v11 = v48;
          sub_6E1BF0(v48);
          goto LABEL_9;
        }
        v23 = v39;
      }
      v9 = (__int64)"reference_wrapper";
      v12 = sub_8D3C80(v23, "reference_wrapper");
      if ( v12 )
      {
        v24 = *(_QWORD *)(*(_QWORD *)(v23 + 168) + 168LL);
        if ( *(_BYTE *)(v24 + 8) )
          goto LABEL_8;
        v25 = *(_QWORD *)(v24 + 32);
        v40 = v25;
        if ( !v25 )
          goto LABEL_8;
        v26 = sub_73D6E0(v25);
        if ( !(unsigned int)sub_8D3A70(v26) )
          goto LABEL_8;
        v9 = v42;
        if ( !(unsigned int)sub_8D5DF0(v26) )
          goto LABEL_8;
      }
      else
      {
        v32 = *(_BYTE *)(v23 + 140);
        if ( (unsigned __int8)(v32 - 9) <= 2u || v32 == 2 && (*(_BYTE *)(v23 + 161) & 8) != 0 )
        {
          v46 = 0;
          v35 = sub_68B9A0(v40);
          if ( !v35 )
            goto LABEL_72;
          v9 = 1;
          v41 = v35;
          sub_84EC30(
            7,
            1,
            0,
            1,
            0,
            *(_QWORD *)(v35 + 24) + 8,
            0,
            (__int64)v49,
            dword_4F06650[0],
            0,
            0,
            (__int64)&v54,
            0,
            0,
            (__int64)&v46);
          sub_6E1990(v41);
          if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) != 0 || !v55 )
            goto LABEL_72;
          v36 = *(_BYTE *)(v54.m128i_i64[0] + 140);
          v40 = v54.m128i_i64[0];
          if ( v36 == 12 )
          {
            v37 = v54.m128i_i64[0];
            do
            {
              v37 = *(_QWORD *)(v37 + 160);
              v38 = *(_BYTE *)(v37 + 140);
            }
            while ( v38 == 12 );
            if ( !v38 )
              goto LABEL_72;
            v33 = v54.m128i_i64[0];
            do
              v33 = *(_QWORD *)(v33 + 160);
            while ( *(_BYTE *)(v33 + 140) == 12 );
          }
          else
          {
            if ( !v36 )
              goto LABEL_72;
            v33 = v54.m128i_i64[0];
          }
          v34 = *(_BYTE *)(v33 + 140);
        }
        else
        {
          if ( !(unsigned int)sub_8D2E30(v23) )
            goto LABEL_8;
          v40 = sub_8D46C0(v23);
          v33 = v40;
          v34 = *(_BYTE *)(v40 + 140);
          if ( v34 == 12 )
          {
            do
            {
              v33 = *(_QWORD *)(v33 + 160);
              v34 = *(_BYTE *)(v33 + 140);
            }
            while ( v34 == 12 );
          }
          else
          {
            v33 = v40;
          }
        }
        if ( (unsigned __int8)(v34 - 9) > 2u )
          goto LABEL_8;
        v9 = v42;
        if ( !(unsigned int)sub_8D5DF0(v33) )
          goto LABEL_8;
      }
      goto LABEL_45;
    case 7:
      goto LABEL_21;
    case 6:
      if ( !(unsigned int)sub_8D2310(*(_QWORD *)(i + 160)) )
      {
        v10 = *(_BYTE *)(i + 140);
        break;
      }
LABEL_21:
      v14 = sub_68B9A0(i);
      goto LABEL_11;
  }
  if ( (unsigned __int8)(v10 - 9) > 2u )
  {
LABEL_8:
    v11 = v48;
    v12 = 0;
    sub_6E1BF0(v48);
    goto LABEL_9;
  }
  v14 = sub_68B9A0(a2);
LABEL_11:
  if ( !v14 )
    goto LABEL_8;
  *(_QWORD *)(qword_4D03C50 + 136LL) = v48;
  if ( v4 )
  {
    while ( 1 )
    {
      v15 = *(_QWORD *)(v4 + 56);
      v16 = sub_68B9A0(v15);
      if ( !v16 )
        break;
      v9 = 0;
      sub_6E1C20(v16, 0, v48);
      v4 = *(_QWORD *)(v4 + 16);
      if ( !v4 )
        goto LABEL_22;
    }
    if ( (!dword_4F077BC || (_DWORD)qword_4F077B4)
      && (unsigned int)sub_8D23B0(v15)
      && !(unsigned int)sub_8D3410(v15)
      && !(unsigned int)sub_8D2600(v15) )
    {
      v9 = v15;
      sub_6E5F60(v4 + 28, v15, 8);
    }
    v12 = 0;
  }
  else
  {
LABEL_22:
    sub_7ADF70(v49, 0);
    dword_4F06648 += 2;
    v17 = sub_7AE2C0(27, dword_4F06648, &dword_4F063F8);
    if ( v50 )
      *v51 = v17;
    else
      v50 = v17;
    v51 = (_QWORD *)v17;
    dword_4F06648 += 2;
    v18 = sub_7AE2C0(28, dword_4F06648, &dword_4F063F8);
    if ( v50 )
      *v51 = v18;
    else
      v50 = v18;
    v51 = (_QWORD *)v18;
    v12 = 0;
    sub_7BC000(v49);
    v9 = (__int64)v53;
    sub_6C0E20(*(_QWORD *)(v14 + 24) + 8LL, v53, 0, &v54);
    if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) == 0 && v55 )
    {
      v20 = v54.m128i_i64[0];
      for ( k = *(unsigned __int8 *)(v54.m128i_i64[0] + 140); (_BYTE)k == 12; k = *(unsigned __int8 *)(v20 + 140) )
        v20 = *(_QWORD *)(v20 + 160);
      v12 = 0;
      if ( (_BYTE)k )
      {
        v12 = 1;
        if ( v55 == 1 && a1 == 113 )
          v12 = sub_731B40(v56, v53, k, v55, v19) == 0;
      }
    }
  }
  sub_6E1BF0(v48);
  v11 = (_BYTE *)v14;
  sub_6E1990(v14);
LABEL_9:
  *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) = (8 * v8)
                                                           | *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7)
                                                           & 0xF7;
  sub_6E2B30(v11, v9);
  sub_6E1DF0(v47);
  return v12;
}
