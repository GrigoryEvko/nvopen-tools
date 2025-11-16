// Function: sub_636A00
// Address: 0x636a00
//
__int64 __fastcall sub_636A00(__int64 *a1, __int64 a2, __m128i *a3, __int64 a4, __int64 *a5)
{
  __int64 v6; // r12
  __int64 *v8; // rdi
  bool v9; // zf
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // rdi
  char v13; // al
  __int64 result; // rax
  char v15; // al
  __int64 v16; // rax
  __int64 v17; // rdi
  int v18; // edx
  unsigned int v19; // edx
  __int64 v20; // rax
  __int64 v21; // rdi
  _QWORD *v22; // rax
  __int16 v23; // ax
  _QWORD *v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int8 v27; // [rsp+4h] [rbp-6Ch]
  unsigned __int64 v29; // [rsp+10h] [rbp-60h]
  unsigned __int64 v31; // [rsp+20h] [rbp-50h]
  __int16 v32; // [rsp+28h] [rbp-48h]
  char v33; // [rsp+2Bh] [rbp-45h]
  int v34; // [rsp+2Ch] [rbp-44h]
  __int64 v35; // [rsp+30h] [rbp-40h] BYREF
  __int64 v36[7]; // [rsp+38h] [rbp-38h] BYREF

  v6 = a2;
  v8 = (__int64 *)*a1;
  v9 = *(_BYTE *)(a2 + 140) == 12;
  v35 = (__int64)v8;
  if ( v9 )
  {
    do
      v6 = *(_QWORD *)(v6 + 160);
    while ( *(_BYTE *)(v6 + 140) == 12 );
  }
  v33 = *((_BYTE *)v8 + 8);
  if ( v33 )
  {
LABEL_6:
    v10 = sub_8D4620(v6);
    v11 = *(_QWORD *)(v6 + 160);
    v29 = v10;
    if ( (a3[2].m128i_i8[8] & 0x40) != 0 )
    {
      v12 = v35;
      *a5 = 0;
    }
    else
    {
      v20 = sub_724D50(10);
      v21 = v35;
      *a5 = v20;
      *(_QWORD *)(v20 + 128) = v6;
      v22 = (_QWORD *)sub_6E1A20(v21);
      v12 = v35;
      *(_QWORD *)(*a5 + 64) = *v22;
      if ( *(_BYTE *)(v12 + 8) != 2 )
      {
        v24 = (_QWORD *)sub_6E1A60(v12);
        v12 = v35;
        *(_QWORD *)(*a5 + 112) = *v24;
      }
      *(_BYTE *)(*a5 + 169) = (32 * (v33 == 1)) | *(_BYTE *)(*a5 + 169) & 0xDF;
    }
    if ( v33 == 1 )
    {
      a4 = v12 + 40;
      v12 = *(_QWORD *)(v12 + 24);
      v13 = (unsigned __int8)a3[2].m128i_i8[9] >> 5;
      v35 = v12;
      v27 = v13 & 1;
    }
    else
    {
      v27 = 0;
      if ( (a3[2].m128i_i8[10] & 1) != 0 )
      {
        if ( (a3[2].m128i_i16[4] & 0x220) == 0 )
        {
          v25 = sub_6E1A20(v12);
          sub_6851C0(2360, v25);
          v12 = v35;
        }
        a3[2].m128i_i8[9] |= 2u;
        v27 = 0;
      }
    }
    v32 = a3[2].m128i_i16[4];
    if ( dword_4F077BC
      && !(_DWORD)qword_4F077B4
      && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) & 0x10) == 0
      && (a3[2].m128i_i8[8] & 0xA0) == 0x80 )
    {
      LOBYTE(v23) = v32 & 0x7F;
      HIBYTE(v23) = ((unsigned __int16)(v32 & 0xFE7F) >> 8) | 1;
      a3[2].m128i_i16[4] = v23;
    }
    if ( v12 )
    {
      if ( !v29 )
        goto LABEL_25;
      v34 = 0;
      v31 = 0;
      do
      {
        while ( 1 )
        {
          sub_634B10(&v35, v11, 0, a3, a4, v36);
          if ( (a3[2].m128i_i8[8] & 0x40) == 0 )
            sub_72A690(v36[0], *a5, 0, 0);
          if ( (a3[2].m128i_i8[9] & 0x20) != 0 )
            break;
          v12 = v35;
          ++v31;
          if ( !v35 )
            goto LABEL_28;
          if ( !v34 && v29 <= v31 )
            goto LABEL_25;
        }
        v34 = 1;
      }
      while ( v35 );
LABEL_28:
      v15 = v34 ^ 1;
    }
    else
    {
      v31 = 0;
      v15 = 1;
    }
    if ( v29 <= v31 || !v15 || (a3[2].m128i_i8[9] |= 0x10u, (a3[2].m128i_i8[8] & 0x40) != 0) )
    {
      v12 = 0;
    }
    else
    {
      v12 = 0;
      *(_BYTE *)(*a5 + 170) |= 0x20u;
      *(_BYTE *)(*a5 + 170) |= 0x40u;
    }
LABEL_25:
    a3[2].m128i_i16[4] = v32 & 0x180 | a3[2].m128i_i16[4] & 0xFE7F;
    result = (__int64)a1;
    if ( v33 != 1 )
    {
      *a1 = v12;
      return result;
    }
    v16 = *(_QWORD *)*a1;
    if ( v16 && *(_BYTE *)(v16 + 8) == 3 )
      v16 = sub_6BBB10(*a1);
    v17 = v35;
    *a1 = v16;
    if ( v17 )
    {
      if ( (a3[2].m128i_i8[8] & 0x20) != 0 )
      {
        v18 = a3[2].m128i_u8[9];
        if ( dword_4F077BC )
        {
          v18 |= 2u;
          a3[2].m128i_i8[9] = v18;
        }
        goto LABEL_39;
      }
      v26 = sub_6E1A20(v17);
      sub_684AA0(dword_4F077BC == 0 ? 5 : 8, dword_4F077BC == 0 ? 1162 : 146, v26);
    }
    v18 = a3[2].m128i_u8[9];
LABEL_39:
    v19 = v18 & 0xFFFFFFDF;
    a3[2].m128i_i8[9] = v19 | (32 * v27);
    return v19 | (32 * v27);
  }
  if ( !(unsigned int)sub_696590(v8, v6) )
  {
    v33 = *(_BYTE *)(v35 + 8);
    goto LABEL_6;
  }
  return sub_631120(a1, v6, a3, (__int64)a5);
}
