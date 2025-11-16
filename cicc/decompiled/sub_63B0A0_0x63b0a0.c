// Function: sub_63B0A0
// Address: 0x63b0a0
//
__int64 __fastcall sub_63B0A0(_BYTE *a1)
{
  __m128i *v1; // r11
  _QWORD *v3; // rcx
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // r13
  char v10; // r12
  bool v11; // r12
  __int64 v12; // r15
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rcx
  char v16; // r12
  bool v17; // zf
  __int64 result; // rax
  __int64 v19; // rax
  int v20; // eax
  __m128i *v21; // r11
  int v22; // eax
  int v23; // eax
  char v24; // al
  __m128i *v25; // [rsp+0h] [rbp-70h]
  __m128i *v26; // [rsp+8h] [rbp-68h]
  __int64 v27; // [rsp+8h] [rbp-68h]
  __int64 v28; // [rsp+10h] [rbp-60h]
  __int64 v29; // [rsp+18h] [rbp-58h]
  __int64 v30; // [rsp+20h] [rbp-50h]
  char v31; // [rsp+2Fh] [rbp-41h]
  __int64 v32; // [rsp+30h] [rbp-40h] BYREF
  _QWORD v33[7]; // [rsp+38h] [rbp-38h] BYREF

  v1 = (__m128i *)(a1 + 136);
  v3 = &qword_4F06BC0;
  v4 = qword_4CFDE50;
  v30 = qword_4CFDE58;
  v28 = qword_4CFDE50;
  v5 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v6 = *(_QWORD *)(v5 + 208);
  v7 = *(unsigned __int8 *)(v5 + 11) | 0xFFFFFF80;
  v31 = *(_BYTE *)(v5 + 11) >> 7;
  v8 = qword_4F06BC0;
  v29 = qword_4F06BC0;
  *(_BYTE *)(v5 + 11) |= 0x80u;
  qword_4CFDE50 = 0;
  v9 = *(_QWORD *)(*(_QWORD *)v6 + 96LL);
  v10 = *(_BYTE *)(v9 + 183);
  *(_BYTE *)(v9 + 183) = v10 | 0x10;
  v11 = (v10 & 0x10) != 0;
  qword_4F06BC0 = *(_QWORD *)(unk_4F07288 + 88LL);
  a1[176] |= 8u;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 80LL) == 8 )
  {
    v12 = *(_QWORD *)(*(_QWORD *)a1 + 88LL);
    qword_4CFDE58 = v12;
    v13 = *(_QWORD *)(v12 + 120);
  }
  else
  {
    v26 = v1;
    v12 = 0;
    v19 = sub_72C930(v4);
    v1 = v26;
    v13 = v19;
  }
  v33[0] = *(_QWORD *)&dword_4F063F8;
  switch ( word_4F06418[0] )
  {
    case 0x38u:
      v25 = v1;
      sub_7B8B50(v13, v8, v7, v3);
      if ( word_4F06418[0] == 73 )
      {
        a1[178] |= 2u;
        if ( dword_4D04964 )
          a1[176] |= 0x80u;
        else
          a1[177] |= 1u;
        sub_637180(v13, 0, v25, 0, dword_4D048B8, 0, v33);
      }
      else
      {
        v32 = v13;
        v27 = sub_6BB940(a1, 0, 0);
        v20 = sub_8D3880(v13);
        v21 = v25;
        if ( v20
          && (v22 = sub_8D3880(v32), v21 = v25, v22)
          && (v23 = sub_6320D0(v27, &v32, (__int64)v25, (__int64)v25), v21 = v25, v23) )
        {
          if ( !*((_QWORD *)a1 + 18) )
            sub_630880(v25->m128i_i64, 0);
        }
        else
        {
          sub_694AA0(v27, v32, 0, 1, v21);
        }
        sub_6E1990(v27);
      }
      break;
    case 0x49u:
      *((_DWORD *)a1 + 44) |= 0x20001u;
      if ( dword_4D04964 )
        a1[176] |= 0x80u;
      else
        a1[177] |= 1u;
      sub_637180(v13, 0, v1, 0, 0, 0, v33);
      break;
    case 0x10u:
      *((_QWORD *)a1 + 18) = sub_72C9D0(v13, v8, v7);
      break;
    case 0x13u:
      *((_QWORD *)a1 + 18) = sub_8250F0(v13, &unk_4F04DA0);
      sub_7B8B50(v13, &unk_4F04DA0, v14, v15);
      break;
    default:
      sub_721090(v13);
  }
  if ( v12 )
  {
    if ( *((_QWORD *)a1 + 18) )
    {
      if ( !*(_QWORD *)(v12 + 152) )
      {
        v24 = (a1[176] << 7) | *(_BYTE *)(v12 + 145) & 0x7F;
        *(_BYTE *)(v12 + 145) = v24;
        *(_QWORD *)(v12 + 152) = *((_QWORD *)a1 + 18);
        *(_BYTE *)(v12 + 145) = (4 * a1[176]) & 0x40 | v24 & 0xBF;
        if ( (char)a1[178] < 0 )
          *(_BYTE *)(v12 + 146) |= 1u;
        *(_QWORD *)(v12 + 192) = v33[0];
        *(_QWORD *)(v12 + 200) = unk_4F061D8;
      }
    }
    else
    {
      *(_BYTE *)(v12 + 145) &= ~0x20u;
    }
  }
  qword_4F06BC0 = v29;
  *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 11) = (v31 << 7)
                                                            | *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 11)
                                                            & 0x7F;
  v16 = *(_BYTE *)(v9 + 183) & 0xEF | (16 * v11);
  v17 = (*(_DWORD *)(v9 + 100))-- == 1;
  *(_BYTE *)(v9 + 183) = v16;
  qword_4CFDE58 = v30;
  result = v28;
  qword_4CFDE50 = v28;
  if ( v17 )
    return sub_6018B0(v6);
  return result;
}
