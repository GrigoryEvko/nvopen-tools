// Function: sub_638440
// Address: 0x638440
//
__int64 __fastcall sub_638440(__int64 *a1, __int64 *a2, __m128i *a3, __int64 a4, int a5)
{
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rcx
  bool v11; // r12
  __int64 v12; // rax
  char v13; // al
  __int64 i; // r15
  __int64 *v15; // rax
  __int64 v16; // rax
  char v17; // dl
  __int64 v18; // rbx
  __int64 v19; // rdi
  __int8 v20; // r10
  int v22; // eax
  int v23; // r15d
  char v24; // dl
  __int64 v25; // rax
  __int64 v26; // rdi
  _QWORD *v27; // rax
  __int64 v28; // rdi
  char j; // al
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rax
  int v33; // eax
  __int64 k; // rax
  _QWORD *v36; // [rsp+8h] [rbp-58h]
  __int64 v37; // [rsp+10h] [rbp-50h]
  __int64 *v38[2]; // [rsp+18h] [rbp-48h] BYREF
  __int64 v39[7]; // [rsp+28h] [rbp-38h] BYREF

  v38[0] = a1;
  v8 = sub_6E1A20(a1);
  v9 = *a2;
  v10 = a3[2].m128i_i64[0];
  a3->m128i_i64[0] = 0;
  v36 = (_QWORD *)v8;
  LOBYTE(v8) = a3[2].m128i_i8[8];
  v39[0] = v9;
  v37 = v10;
  a3[2].m128i_i8[8] = v8 & 0xF7;
  a3[2].m128i_i64[0] = a4;
  v11 = (v8 & 8) != 0;
  v12 = unk_4D03C50;
  a3->m128i_i64[1] = 0;
  if ( (*(_BYTE *)(v12 + 21) & 0x10) != 0 )
  {
    a3[2].m128i_i16[4] &= 0xFE7Fu;
    goto LABEL_3;
  }
  if ( dword_4D04964 )
    goto LABEL_7;
  if ( a4 || !a3[1].m128i_i64[0] )
  {
    if ( !dword_4F077BC )
    {
      if ( !(_DWORD)qword_4F077B4 )
        goto LABEL_32;
      goto LABEL_7;
    }
    if ( (_DWORD)qword_4F077B4 )
      goto LABEL_7;
    v15 = *(__int64 **)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 624);
    if ( v15 )
    {
      v16 = *v15;
      if ( v16 )
      {
        v17 = *(_BYTE *)(v16 + 80);
        if ( v17 == 9 || v17 == 7 )
        {
          v32 = *(_QWORD *)(v16 + 88);
        }
        else
        {
          if ( v17 != 21 )
            goto LABEL_16;
          v32 = *(_QWORD *)(*(_QWORD *)(v16 + 88) + 192LL);
        }
        if ( v32 )
          goto LABEL_32;
      }
    }
LABEL_16:
    if ( a3[2].m128i_i8[11] < 0 )
      goto LABEL_32;
LABEL_7:
    a3[2].m128i_i8[8] |= 0x80u;
    goto LABEL_3;
  }
LABEL_32:
  a3[2].m128i_i8[9] |= 1u;
LABEL_3:
  v13 = *(_BYTE *)(v9 + 140);
  for ( i = v9; v13 == 12; v13 = *(_BYTE *)(i + 140) )
    i = *(_QWORD *)(i + 160);
  v39[0] = i;
  switch ( v13 )
  {
    case 0:
    case 14:
      a4 = 0;
      sub_6319F0(v38[0], v9, a3, a3->m128i_i64);
      goto LABEL_26;
    case 5:
      if ( (!dword_4F077BC || qword_4F077A8 <= 0x9EFBu) && !(_DWORD)qword_4F077B4 )
        goto LABEL_46;
      v27 = (_QWORD *)v38[0][3];
      if ( !v27 || !*v27 )
        goto LABEL_46;
      sub_636FC0((__int64 *)v38, i, a3, a3->m128i_i64);
      a4 = 0;
LABEL_26:
      v20 = a3[2].m128i_i8[8] & 0xF7 | (8 * v11);
      a3[2].m128i_i8[8] = v20;
      if ( (v20 & 0x40) == 0 && !a3->m128i_i64[1] )
        sub_630880(a3->m128i_i64, a4);
      if ( (a3[2].m128i_i8[9] & 0xA) == 8 )
      {
        if ( (a3[2].m128i_i8[8] & 0x20) != 0 )
          a3[2].m128i_i8[9] |= 2u;
        else
          sub_6851C0(517, v36);
      }
      a3[2].m128i_i64[0] = v37;
      return v37;
    case 8:
      v22 = sub_8D23E0(i);
      a3[2].m128i_i8[10] |= 8u;
      v23 = v22;
      sub_635980(v38, v39, a3, v36, a3->m128i_i64);
      v24 = *(_BYTE *)(v39[0] + 140);
      if ( v24 == 12 )
      {
        v25 = v39[0];
        do
        {
          v25 = *(_QWORD *)(v25 + 160);
          v24 = *(_BYTE *)(v25 + 140);
        }
        while ( v24 == 12 );
      }
      if ( v24 )
      {
        v28 = sub_8D40F0(v39[0]);
        for ( j = *(_BYTE *)(v28 + 140); j == 12; j = *(_BYTE *)(v28 + 140) )
          v28 = *(_QWORD *)(v28 + 160);
        a4 = 0;
        if ( a5 && (unsigned __int8)(j - 9) <= 2u )
          a4 = sub_630000(v28, a3, (int)v36);
        if ( v23 && a3->m128i_i64[0] )
        {
          v30 = v39[0];
          *(_QWORD *)(a3->m128i_i64[0] + 128) = v39[0];
          if ( (a3[2].m128i_i8[10] & 4) != 0 )
          {
            a3[2].m128i_i8[9] |= 0x10u;
          }
          else
          {
            if ( HIDWORD(qword_4F077B4) )
              goto LABEL_65;
            v33 = sub_8D3410(v30);
            v30 = v39[0];
            if ( !v33 )
              goto LABEL_65;
            for ( k = v39[0]; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
              ;
            if ( (*(_BYTE *)(k + 169) & 0x20) != 0 )
            {
              if ( (a3[2].m128i_i8[8] & 0x20) == 0 )
              {
                v30 = 1345;
                sub_6851C0(1345, v36);
              }
              a3[2].m128i_i8[9] |= 2u;
              *a2 = sub_72C930(v30);
            }
            else
            {
LABEL_65:
              *a2 = v30;
            }
          }
        }
      }
      else
      {
        a3[2].m128i_i8[9] |= 2u;
        a4 = 0;
      }
      goto LABEL_26;
    case 9:
    case 10:
    case 11:
      if ( (*(_BYTE *)(i + 177) & 0x20) != 0
        || (dword_4F04C44 != -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0)
        && (i = v39[0], (unsigned int)sub_82ED80(v38[0])) )
      {
        a4 = 0;
        sub_6319F0(v38[0], i, a3, a3->m128i_i64);
      }
      else
      {
        v18 = 0;
        if ( a5 )
        {
          v31 = sub_630000(i, a3, (int)v36);
          i = v39[0];
          v18 = v31;
        }
        sub_6333F0((__int64 *)v38, i, a3, (__int64)v36, a3->m128i_i64);
        if ( a4 )
        {
          v19 = a4;
          a4 = v18;
          sub_694F90(v19);
        }
        else
        {
          a4 = v18;
        }
      }
      goto LABEL_26;
    case 15:
      sub_636A00((__int64 *)v38, i, a3, (__int64)v36, a3->m128i_i64);
      if ( a4 )
      {
        v26 = a4;
        a4 = 0;
        sub_694F90(v26);
      }
      goto LABEL_26;
    default:
LABEL_46:
      sub_721090(a1);
  }
}
