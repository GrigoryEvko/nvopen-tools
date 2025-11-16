// Function: sub_631120
// Address: 0x631120
//
__int64 __fastcall sub_631120(__int64 *a1, __int64 a2, const __m128i *a3, __int64 a4)
{
  __int64 v6; // r15
  _BOOL4 v7; // r13d
  __int8 v8; // al
  char v9; // al
  __int64 v10; // r14
  char v11; // dl
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int8 v17; // al
  int v18; // eax
  __int64 result; // rax
  __int64 v20; // rsi
  __int64 i; // r14
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // r9
  __int64 v26; // r14
  __int64 v27; // rdi
  char v28; // dl
  unsigned int v29; // ecx
  __int64 v30; // r15
  __int64 v31; // r15
  char *v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rdx
  __int64 v35; // rdi
  __int64 v36; // r8
  __int64 v37; // r15
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  unsigned int v42; // [rsp+4h] [rbp-ACh]
  unsigned int v43; // [rsp+4h] [rbp-ACh]
  unsigned int v44; // [rsp+4h] [rbp-ACh]
  __int64 v45; // [rsp+8h] [rbp-A8h]
  __int64 v46; // [rsp+8h] [rbp-A8h]
  __int64 v47; // [rsp+8h] [rbp-A8h]
  unsigned int v48; // [rsp+8h] [rbp-A8h]
  __int64 v49; // [rsp+10h] [rbp-A0h]
  __int64 v50; // [rsp+10h] [rbp-A0h]
  __int64 *v51; // [rsp+18h] [rbp-98h]
  __int64 v54; // [rsp+38h] [rbp-78h] BYREF
  _QWORD v55[2]; // [rsp+40h] [rbp-70h] BYREF
  _OWORD v56[2]; // [rsp+50h] [rbp-60h] BYREF
  __m128i v57; // [rsp+70h] [rbp-40h]

  v51 = (__int64 *)*a1;
  v6 = *a1;
  v7 = *(_BYTE *)(*a1 + 8) == 1;
  do
  {
    if ( !v7 )
      goto LABEL_3;
    for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    if ( (unsigned int)sub_8D2FB0(i) )
      goto LABEL_3;
    if ( (unsigned __int8)(*(_BYTE *)(i + 140) - 9) <= 2u )
    {
      if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 178LL) & 0x40) == 0 && !dword_4D04428 )
      {
        v22 = sub_6E1A20(v6);
        sub_685360(285, v22);
      }
      goto LABEL_3;
    }
    if ( !*(_QWORD *)(v6 + 24) )
    {
      if ( !dword_4D04428 )
        sub_6851C0(29, v6 + 40);
      goto LABEL_3;
    }
    v23 = sub_6E1A20(v6);
    v6 = *(_QWORD *)(v6 + 24);
    v49 = v23;
    v24 = v6;
    while ( 1 )
    {
      if ( !v24 )
        goto LABEL_48;
      v25 = *(_QWORD *)v24;
      if ( *(_BYTE *)(v24 + 8) != 2 )
        break;
      if ( !v25 )
        goto LABEL_48;
      if ( *(_BYTE *)(v25 + 8) == 3 )
        v24 = sub_6BBB10(v24);
      else
        v24 = *(_QWORD *)v24;
    }
    if ( !v25 || *(_BYTE *)(v25 + 8) == 3 && (v25 = sub_6BBB10(v24)) == 0 )
    {
LABEL_48:
      v26 = 0;
      goto LABEL_49;
    }
    v26 = sub_6E1A20(v25);
LABEL_49:
    if ( (dword_4F077C4 != 2 || unk_4F07778 > 201102 || dword_4F07774) && !qword_4F077B4 )
    {
      if ( *(_BYTE *)(v6 + 8) != 1 )
        goto LABEL_52;
LABEL_95:
      v29 = dword_4F077C0 == 0 ? 8 : 5;
LABEL_96:
      v34 = v6;
      while ( 1 )
      {
        v6 = v34;
        v34 = *(_QWORD *)(v34 + 24);
        if ( !v34 )
          break;
        v35 = v34;
        while ( v35 )
        {
          v36 = *(_QWORD *)v35;
          if ( *(_BYTE *)(v35 + 8) != 2 )
          {
            if ( v36 )
            {
              if ( *(_BYTE *)(v36 + 8) != 3
                || (v44 = v29, v47 = v34, v41 = sub_6BBB10(v35), v34 = v47, v29 = v44, (v36 = v41) != 0) )
              {
                v42 = v29;
                v45 = v34;
                v39 = sub_6E1A20(v36);
                v34 = v45;
                v29 = v42;
                v26 = v39;
              }
            }
            break;
          }
          if ( !v36 )
            break;
          if ( *(_BYTE *)(v36 + 8) == 3 )
          {
            v43 = v29;
            v46 = v34;
            v40 = sub_6BBB10(v35);
            v34 = v46;
            v29 = v43;
            v35 = v40;
          }
          else
          {
            v35 = *(_QWORD *)v35;
          }
        }
        if ( *(_BYTE *)(v34 + 8) != 1 )
        {
          v6 = v34;
          goto LABEL_72;
        }
      }
      if ( *(_BYTE *)(v6 + 8) == 1 && !dword_4D04428 )
      {
        v48 = v29;
        sub_6851C0(29, v6 + 40);
        v29 = v48;
      }
      goto LABEL_72;
    }
    v28 = *(_BYTE *)(v6 + 8);
    if ( dword_4D04964 )
    {
      if ( v28 == 1 )
      {
        v29 = 8;
        if ( dword_4F077C0 )
        {
          v29 = byte_4F07472[0];
          if ( byte_4F07472[0] == 3 )
            v29 = 5;
        }
        goto LABEL_96;
      }
      v29 = byte_4F07472[0];
    }
    else
    {
      v29 = 5;
      if ( v28 == 1 )
        goto LABEL_95;
    }
LABEL_72:
    if ( (a3[2].m128i_i8[8] & 0x20) != 0 )
      a3[2].m128i_i8[9] = (2 * (sub_67D3C0(991, v29, v49) & 1)) | a3[2].m128i_i8[9] & 0xFD;
    else
      sub_684AA0(v29, 991, v49);
LABEL_52:
    if ( v26 )
    {
      if ( dword_4F077C0 )
      {
        v27 = *(_QWORD *)v6;
        if ( *(_QWORD *)v6 && *(_BYTE *)(v27 + 8) == 3 )
          v27 = sub_6BBB10(v6);
        if ( !sub_62F7E0(v27) )
          sub_684B30(1162, v26);
      }
      else
      {
        if ( (a3[2].m128i_i8[8] & 0x20) != 0 )
        {
          v8 = a3[2].m128i_i8[9] | 2;
          a3[2].m128i_i8[9] = v8;
          if ( *(_BYTE *)(v6 + 8) != 2 )
            break;
          goto LABEL_6;
        }
        sub_6851C0(146, v26);
      }
    }
LABEL_3:
    if ( *(_BYTE *)(v6 + 8) != 2 )
      break;
    if ( (a3[2].m128i_i8[8] & 0x20) == 0 )
    {
      v20 = sub_6E1A20(v6);
      sub_685360(2357, v20);
      goto LABEL_7;
    }
    v8 = a3[2].m128i_i8[9];
LABEL_6:
    a3[2].m128i_i8[9] = v8 | 2;
LABEL_7:
    a2 = sub_72C930();
    while ( 1 )
    {
      if ( !v6 )
LABEL_136:
        BUG();
      v9 = *(_BYTE *)(v6 + 8);
      if ( v9 != 2 )
        break;
      if ( !*(_QWORD *)v6 )
        goto LABEL_136;
      if ( *(_BYTE *)(*(_QWORD *)v6 + 8LL) == 3 )
        v6 = sub_6BBB10(v6);
      else
        v6 = *(_QWORD *)v6;
    }
  }
  while ( v9 );
  v10 = v6;
  v56[0] = _mm_loadu_si128(a3);
  v56[1] = _mm_loadu_si128(a3 + 1);
  v57 = _mm_loadu_si128(a3 + 2);
  if ( !dword_4F077BC || (_DWORD)qword_4F077B4 || (v11 = 1, (a3[2].m128i_i8[11] & 0x40) == 0) )
    v11 = a3[2].m128i_i8[12] & 1;
  v57.m128i_i64[1] = v11 & 1 | v57.m128i_i64[1] & 0xFFFFFFFEFFFFFFFELL;
  if ( word_4D04898 && (v57.m128i_i8[8] & 4) != 0 )
    v57.m128i_i8[8] &= ~4u;
  v12 = v6;
  sub_694AA0(v6, a2, 0, dword_4D048B8, v56);
  if ( (v57.m128i_i8[9] & 2) != 0 || (v12 = v6, (unsigned int)sub_6E1A80(v6)) )
  {
    a3[2].m128i_i8[9] |= 2u;
  }
  else if ( (a3[2].m128i_i32[2] & 0x40000020) == 0x40000000 )
  {
    if ( *((_QWORD *)&v56[0] + 1) )
    {
      if ( *(_BYTE *)(*((_QWORD *)&v56[0] + 1) + 48LL) == 5 )
      {
        v32 = *(char **)(*((_QWORD *)&v56[0] + 1) + 56LL);
        if ( v32 )
        {
          if ( v32[193] < 0 )
          {
            v50 = *(_QWORD *)v32;
            v12 = 3152;
            v33 = sub_6E1A20(v6);
            sub_685490(3152, v33, v50);
          }
        }
      }
    }
  }
  v17 = a3[2].m128i_i8[8];
  if ( v57.m128i_i8[10] < 0 && (a3[2].m128i_i8[10] |= 0x80u, word_4D04898) && (v17 & 4) != 0 )
  {
    if ( (a3[2].m128i_i16[4] & 0x220) == 0 && *((_QWORD *)&v56[0] + 1) )
    {
      v54 = sub_724DC0(v12, word_4D04898, v13, v14, v15, v16);
      v30 = sub_6E1A20(v6);
      v55[0] = 0;
      v55[1] = 0;
      if ( (unsigned int)sub_7A1C60(DWORD2(v56[0]), v30, a2, 1, v54, (unsigned int)v55, 0) )
      {
        sub_67E3D0(v55);
      }
      else
      {
        v31 = sub_67D9D0(28, v30);
        sub_67E370(v31, v55);
        sub_685910(v31);
      }
      sub_724E30(&v54);
      v17 = a3[2].m128i_i8[8];
    }
    v18 = v17 & 0x40;
    if ( !v18 )
    {
      *((_QWORD *)&v56[0] + 1) = 0;
      *(_QWORD *)&v56[0] = sub_72C9A0();
      LOBYTE(v18) = a3[2].m128i_i8[8] & 0x40;
    }
    a3[2].m128i_i8[9] |= 2u;
  }
  else
  {
    LOBYTE(v18) = v17 & 0x40;
  }
  if ( (_BYTE)v18 )
  {
    *(_QWORD *)a4 = 0;
    if ( v7 )
      goto LABEL_28;
  }
  else
  {
    if ( (a3[2].m128i_i8[9] & 2) != 0 )
    {
      *(_QWORD *)a4 = sub_72C9A0();
    }
    else if ( *(_QWORD *)&v56[0] )
    {
      *(_QWORD *)a4 = *(_QWORD *)&v56[0];
    }
    else
    {
      v37 = *((_QWORD *)&v56[0] + 1);
      if ( *((_QWORD *)&v56[0] + 1) )
      {
        v38 = sub_724D50(9);
        *(_QWORD *)a4 = v38;
        *(_QWORD *)(v38 + 176) = v37;
        *(_QWORD *)(*(_QWORD *)a4 + 128LL) = a2;
        *(_QWORD *)(*(_QWORD *)a4 + 64LL) = *(_QWORD *)sub_6E1A20(v10);
        if ( *(_BYTE *)(v10 + 8) != 2 )
          *(_QWORD *)(*(_QWORD *)a4 + 112LL) = *(_QWORD *)sub_6E1A60(v10);
        if ( (*(_BYTE *)(v37 + 48) & 0xFB) == 2 )
          *(_BYTE *)(*(_QWORD *)a4 + 170LL) = *(_BYTE *)(*(_QWORD *)(v37 + 56) + 170LL) & 2
                                            | *(_BYTE *)(*(_QWORD *)a4 + 170LL) & 0xFD;
        a3[2].m128i_i8[9] |= 4u;
        if ( *(_QWORD *)(v37 + 16) )
          sub_734250(v37, (((unsigned __int8)v57.m128i_i8[10] >> 4) ^ 1) & 1);
      }
    }
    if ( v7 )
    {
LABEL_28:
      result = *v51;
      if ( *v51 && *(_BYTE *)(result + 8) == 3 )
        result = sub_6BBB10(v51);
      goto LABEL_31;
    }
  }
  result = *(_QWORD *)v10;
  if ( *(_QWORD *)v10 && *(_BYTE *)(result + 8) == 3 )
    result = sub_6BBB10(v10);
LABEL_31:
  *a1 = result;
  return result;
}
