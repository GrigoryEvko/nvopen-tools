// Function: sub_65D790
// Address: 0x65d790
//
__int64 __fastcall sub_65D790(__int64 a1, _QWORD *a2, char a3)
{
  char v4; // bl
  _QWORD *v5; // rbx
  __int64 result; // rax
  __int64 v7; // r14
  __int64 v8; // rax
  _QWORD *v9; // r13
  int v10; // ecx
  __int64 *v11; // r8
  unsigned int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 *v15; // rsi
  _DWORD *v16; // rdx
  _DWORD *v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rdi
  char v20; // dl
  __int64 v21; // rax
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // r15
  __int64 v25; // r15
  __int64 i; // rax
  __int64 v27; // r14
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // rax
  __int64 v33; // r15
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // rcx
  int v38; // eax
  __int64 v39; // [rsp+0h] [rbp-2E0h]
  __int64 *v40; // [rsp+0h] [rbp-2E0h]
  int v41; // [rsp+0h] [rbp-2E0h]
  __int64 v42; // [rsp+8h] [rbp-2D8h]
  _QWORD *v43; // [rsp+8h] [rbp-2D8h]
  __int64 *v44; // [rsp+8h] [rbp-2D8h]
  __int64 v45; // [rsp+8h] [rbp-2D8h]
  char v46; // [rsp+1Ch] [rbp-2C4h] BYREF
  __int64 v47; // [rsp+20h] [rbp-2C0h] BYREF
  __int64 v48; // [rsp+28h] [rbp-2B8h] BYREF
  _QWORD v49[2]; // [rsp+30h] [rbp-2B0h] BYREF
  __m128i v50; // [rsp+40h] [rbp-2A0h]
  __m128i v51; // [rsp+50h] [rbp-290h]
  __m128i v52; // [rsp+60h] [rbp-280h]
  _QWORD v53[12]; // [rsp+70h] [rbp-270h] BYREF
  _QWORD v54[66]; // [rsp+D0h] [rbp-210h] BYREF

  v4 = a3 & 1;
  sub_8600D0(2, 0xFFFFFFFFLL, 0, 0);
  *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 8) = v4
                                                           | *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 8)
                                                           & 0xFE;
  v5 = (_QWORD *)sub_726810();
  sub_7335B0(v5);
  v5[1] = *a2;
  if ( !(unsigned int)sub_7BE280(27, 125, 0, 0) )
    goto LABEL_2;
  memset(v54, 0, 0x1D8u);
  v54[19] = v54;
  v54[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v54[22]) |= 1u;
  v47 = *(_QWORD *)&dword_4F063F8;
  v54[23] = sub_5CC190(1);
  if ( word_4F06418[0] == 76 )
  {
    v7 = 0;
    sub_6446A0(&v54[23], 8u);
    sub_7B8B50(&v54[23], 8, v36, v37);
    goto LABEL_10;
  }
  if ( word_4F06418[0] != 1 && !(unsigned int)sub_651B00(2u) )
  {
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    sub_6851D0(531);
    v7 = sub_72C930(531);
    v49[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
    v50 = _mm_loadu_si128(&xmmword_4F06660[1]);
    v50.m128i_i8[1] |= 0x20u;
    v51 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v49[1] = *(_QWORD *)dword_4F07508;
    v52 = _mm_loadu_si128(&xmmword_4F06660[3]);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    goto LABEL_10;
  }
  v15 = v54;
  v54[15] = v54[15] & 0xFFFFFFBFF7FFFFFFLL | ((unsigned __int64)(word_4D04430 & 1) << 27);
  memset(v53, 0, 0x58u);
  sub_672A20(18, v54, v53);
  if ( (v54[1] & 0x20) != 0 )
  {
    v15 = &v47;
    sub_6851C0(255, &v47);
  }
  else if ( (v54[1] & 0x100LL) != 0 )
  {
    v15 = &v47;
    sub_6851C0(531, &v47);
    v54[34] = sub_72C930(531);
    v54[35] = v54[34];
    v54[36] = v54[34];
  }
  else if ( (v54[1] & 1) == 0 )
  {
    v15 = (__int64 *)v54[34];
    sub_64E990((__int64)&dword_4F063F8, v54[34], 0, 0, 0, 1);
  }
  if ( word_4F06418[0] != 1 )
  {
    if ( word_4F06418[0] == 34 || word_4F06418[0] == 27 )
      goto LABEL_32;
    v16 = &dword_4F077C4;
    if ( dword_4F077C4 != 2 )
    {
      if ( word_4F06418[0] != 25 )
        goto LABEL_33;
      goto LABEL_32;
    }
    if ( dword_4D04474 && word_4F06418[0] == 52
      || dword_4D0485C && word_4F06418[0] == 25
      || word_4F06418[0] == 156
      || ((word_4F06418[0] - 25) & 0xFFF7) == 0 )
    {
LABEL_32:
      v15 = v54;
      sub_626F50((-(__int64)(unk_4D047EC == 0) & 0xFFFFFFFFFFFFC000LL) + 16387, (__int64)v54, 0, (__int64)v49, 0, v53);
      goto LABEL_33;
    }
    goto LABEL_76;
  }
  if ( dword_4F077C4 != 2 )
    goto LABEL_32;
  if ( (unk_4D04A11 & 2) != 0 )
  {
    if ( (unk_4D04A12 & 1) == 0 )
      goto LABEL_32;
LABEL_75:
    if ( (unk_4D04A11 & 2) != 0 )
      goto LABEL_33;
    goto LABEL_76;
  }
  v15 = 0;
  if ( !(unsigned int)sub_7C0F00(0, 0) || (unk_4D04A12 & 1) == 0 || word_4F06418[0] == 25 )
    goto LABEL_32;
  v17 = &dword_4F077C4;
  if ( dword_4F077C4 != 2 )
    goto LABEL_33;
  if ( word_4F06418[0] == 1 )
    goto LABEL_75;
LABEL_76:
  v15 = 0;
  if ( !(unsigned int)sub_7C0F00(0, 0) && word_4F06418[0] == 15 )
    goto LABEL_32;
LABEL_33:
  v19 = (__int64)v54;
  sub_65C470((__int64)v54, (__int64)v15, (__int64)v16, (__int64)v17, v18);
  if ( (v54[15] & 0x2000000000LL) != 0 )
  {
    v19 = (__int64)v54;
    sub_6451E0((__int64)v54);
  }
  if ( !dword_4D048B8 )
    goto LABEL_44;
  v20 = *(_BYTE *)(v54[36] + 140LL);
  if ( v20 == 12 )
  {
    v21 = v54[36];
    do
    {
      v21 = *(_QWORD *)(v21 + 160);
      v20 = *(_BYTE *)(v21 + 140);
    }
    while ( v20 == 12 );
  }
  if ( !v20 )
    goto LABEL_45;
  if ( dword_4F077C4 == 2 )
  {
    v45 = v54[36];
    if ( (unsigned int)sub_8D23B0(v54[36]) )
      sub_8AE000(v45);
  }
  sub_645520(&v54[36]);
  v22 = v54[36];
  if ( unk_4D047EC && (unsigned int)sub_8DD010(v54[36]) )
  {
    v19 = 975;
    sub_6851C0(975, &v47);
  }
  else
  {
    if ( !(unsigned int)sub_8D23B0(v22) )
    {
      if ( (unsigned int)sub_8D3110(v22) )
      {
        v32 = sub_8D46C0(v22);
        if ( !(unsigned int)sub_8D3D40(v32) )
        {
          v19 = 1791;
          sub_6851C0(1791, &v47);
          goto LABEL_44;
        }
      }
      if ( (unsigned int)sub_8D3320(v22) )
      {
        v33 = sub_8D46C0(v22);
        if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v33) )
          sub_8AE000(v33);
        if ( (unsigned int)sub_8D23B0(v33) && !(unsigned int)sub_8D2600(v33) )
          sub_685A50(833, &v47, v33, 7);
      }
      else if ( (unsigned int)sub_8D5830(v22) )
      {
        v19 = 8;
        sub_5EB950(8u, 987, v22, (__int64)&v47);
        goto LABEL_44;
      }
      sub_8DCE90(v54[36]);
      if ( (unsigned int)sub_8D96C0(v54[36]) )
        sub_684B00(534, &v47);
      goto LABEL_45;
    }
    v19 = (unsigned int)sub_67F240(v22);
    sub_685A50(v19, &v47, v22, 8);
  }
LABEL_44:
  v54[34] = sub_72C930(v19);
  v54[35] = v54[34];
  v54[36] = v54[34];
LABEL_45:
  v23 = sub_736000();
  v5[2] = v23;
  v24 = v23;
  if ( (v54[2] & 2) != 0 )
  {
    v34 = sub_885AD0(7, v49, (unsigned int)dword_4F04C5C, 0);
    v54[0] = v34;
    *(_QWORD *)(v34 + 88) = v5[2];
    sub_877D80(v5[2], v34);
    sub_8756F0(3, v54[0], v54[0] + 48LL, v54[44]);
    v35 = v54[0];
    *(_QWORD *)(*(_QWORD *)(v54[0] + 88LL) + 256LL) = v54[35];
    sub_8756B0(v35);
    v24 = v5[2];
  }
  if ( v54[0] )
    sub_729470(v24, v53);
  else
    *(_QWORD *)(v24 + 72) = sub_729420(0, v53);
  *(_BYTE *)(v5[2] + 89LL) |= 1u;
  sub_644830((__int64)v54, 7, v5[2], 1);
  if ( (unsigned int)sub_8D3A70(v54[36]) )
  {
    if ( v54[0] )
      v48 = *(_QWORD *)(v54[0] + 48LL);
    else
      v48 = v54[4];
    v42 = sub_87D180(v54[36], 0, 0, (unsigned int)&v48, v54[36], (unsigned int)&v46, 0);
    v25 = sub_87CF10(v54[36], v54[36], &v48);
    if ( v42 )
    {
      for ( i = *(_QWORD *)(v42 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v39 = v42;
      v43 = **(_QWORD ***)(i + 168);
      v27 = sub_725A70(5);
      *(_QWORD *)(v27 + 56) = v39;
      v28 = sub_73F570(v39, *v43, 0, 1, 1);
      *(_BYTE *)(v27 + 72) |= 1u;
      *(_QWORD *)(v27 + 64) = v28;
    }
    else
    {
      v27 = sub_725A70(7);
    }
    *(_QWORD *)(v27 + 8) = v5[2];
    if ( v25 )
    {
      *(_QWORD *)(v27 + 16) = v25;
      *(_BYTE *)(v25 + 193) |= 0x40u;
    }
  }
  else
  {
    v27 = sub_725A70(7);
    *(_QWORD *)(v27 + 8) = v5[2];
  }
  sub_7340D0(v27, 0, 1);
  v5[4] = v27;
  v7 = v54[36];
  sub_65C470((__int64)v54, 0, v29, v30, v31);
  sub_643EB0((__int64)v54, 0);
LABEL_10:
  v8 = *(_QWORD *)(a1 + 72);
  v9 = *(_QWORD **)(v8 + 16);
  if ( v9 )
  {
    v10 = 0;
    v11 = &v47;
    v12 = dword_4D048B8;
    while ( 1 )
    {
      if ( !v12 || v7 && !*(_BYTE *)(v7 + 140) )
        goto LABEL_20;
      v13 = v9[2];
      if ( v13 )
      {
        if ( !v10 )
        {
          if ( v5[2] )
          {
            v14 = *(_QWORD *)(v13 + 120);
            if ( *(_BYTE *)(v14 + 140) )
            {
              v40 = v11;
              v38 = sub_8E0610(v14, v7);
              v10 = 0;
              v11 = v40;
              if ( v38 )
              {
                sub_685330(533, v40, *(_QWORD *)(v9[2] + 120LL));
                v11 = v40;
                v10 = 1;
              }
            }
          }
        }
LABEL_20:
        if ( !*v9 )
        {
          *v9 = v5;
          goto LABEL_22;
        }
        v12 = dword_4D048B8;
        v9 = (_QWORD *)*v9;
      }
      else
      {
        if ( !*v9 )
        {
          v41 = v10;
          v44 = v11;
          sub_6851C0(532, v11);
          v11 = v44;
          v10 = v41;
          goto LABEL_20;
        }
        v9 = (_QWORD *)*v9;
      }
    }
  }
  *(_QWORD *)(v8 + 16) = v5;
LABEL_22:
  sub_7BE280(28, 18, 0, 0);
LABEL_2:
  result = sub_86FD00(0, 0, 1, 0, 0, 0);
  v5[3] = result;
  *(_QWORD *)(result + 24) = a1;
  return result;
}
