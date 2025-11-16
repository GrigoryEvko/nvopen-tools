// Function: sub_6F3DD0
// Address: 0x6f3dd0
//
void __fastcall sub_6F3DD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // al
  _BYTE *v7; // rax
  char v8; // al
  char v9; // al
  char v10; // al
  _QWORD *v11; // rax
  __int64 *v12; // rax
  __m128i v13; // [rsp+0h] [rbp-170h] BYREF
  __m128i v14; // [rsp+10h] [rbp-160h]
  __m128i v15; // [rsp+20h] [rbp-150h]
  __m128i v16; // [rsp+30h] [rbp-140h]
  __m128i v17; // [rsp+40h] [rbp-130h]
  __m128i v18; // [rsp+50h] [rbp-120h]
  __m128i v19; // [rsp+60h] [rbp-110h]
  __m128i v20; // [rsp+70h] [rbp-100h]
  __m128i v21; // [rsp+80h] [rbp-F0h]
  __m128i v22; // [rsp+90h] [rbp-E0h]
  __m128i v23; // [rsp+A0h] [rbp-D0h]
  __m128i v24; // [rsp+B0h] [rbp-C0h]
  __m128i v25; // [rsp+C0h] [rbp-B0h]
  __m128i v26; // [rsp+D0h] [rbp-A0h]
  __m128i v27; // [rsp+E0h] [rbp-90h]
  __m128i v28; // [rsp+F0h] [rbp-80h]
  __m128i v29; // [rsp+100h] [rbp-70h]
  __m128i v30; // [rsp+110h] [rbp-60h]
  __m128i v31; // [rsp+120h] [rbp-50h]
  __m128i v32; // [rsp+130h] [rbp-40h]
  __m128i v33; // [rsp+140h] [rbp-30h]
  __m128i v34; // [rsp+150h] [rbp-20h]

  v6 = *(_BYTE *)(a1 + 16);
  v13 = _mm_loadu_si128((const __m128i *)a1);
  v14 = _mm_loadu_si128((const __m128i *)(a1 + 16));
  v15 = _mm_loadu_si128((const __m128i *)(a1 + 32));
  v16 = _mm_loadu_si128((const __m128i *)(a1 + 48));
  v17 = _mm_loadu_si128((const __m128i *)(a1 + 64));
  v18 = _mm_loadu_si128((const __m128i *)(a1 + 80));
  v19 = _mm_loadu_si128((const __m128i *)(a1 + 96));
  v20 = _mm_loadu_si128((const __m128i *)(a1 + 112));
  v21 = _mm_loadu_si128((const __m128i *)(a1 + 128));
  if ( v6 == 2 )
  {
    v22 = _mm_loadu_si128((const __m128i *)(a1 + 144));
    v23 = _mm_loadu_si128((const __m128i *)(a1 + 160));
    v24 = _mm_loadu_si128((const __m128i *)(a1 + 176));
    v25 = _mm_loadu_si128((const __m128i *)(a1 + 192));
    v26 = _mm_loadu_si128((const __m128i *)(a1 + 208));
    v27 = _mm_loadu_si128((const __m128i *)(a1 + 224));
    v28 = _mm_loadu_si128((const __m128i *)(a1 + 240));
    v29 = _mm_loadu_si128((const __m128i *)(a1 + 256));
    v30 = _mm_loadu_si128((const __m128i *)(a1 + 272));
    v31 = _mm_loadu_si128((const __m128i *)(a1 + 288));
    v32 = _mm_loadu_si128((const __m128i *)(a1 + 304));
    v33 = _mm_loadu_si128((const __m128i *)(a1 + 320));
    v34 = _mm_loadu_si128((const __m128i *)(a1 + 336));
    goto LABEL_7;
  }
  if ( v6 != 5 && v6 != 1 )
  {
    if ( v6 == 3 )
    {
      v7 = *(_BYTE **)(a1 + 136);
      if ( (v7[81] & 0x10) == 0 )
        *(_BYTE *)(*(_QWORD *)v7 + 73LL) |= 8u;
    }
LABEL_7:
    if ( (_DWORD)a3 )
      goto LABEL_8;
    goto LABEL_20;
  }
  v22.m128i_i64[0] = *(_QWORD *)(a1 + 144);
  if ( (_DWORD)a3 )
  {
LABEL_8:
    v8 = *(_BYTE *)(a1 + 16);
    if ( v8 == 3 )
    {
      sub_6F3BA0((__m128i *)a1, 1);
    }
    else
    {
      if ( v8 == 4 )
        sub_6EE880(a1, 0);
      sub_6F69D0(a1, 0);
    }
    goto LABEL_12;
  }
LABEL_20:
  if ( !(_DWORD)a2 )
  {
    v10 = *(_BYTE *)(a1 + 16);
    if ( v10 != 3 )
      goto LABEL_22;
LABEL_31:
    sub_6F3BA0((__m128i *)a1, 0);
    goto LABEL_12;
  }
  sub_6F8020(
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    v13.m128i_i64[0],
    v13.m128i_i64[1],
    v14.m128i_i64[0],
    v14.m128i_i64[1],
    v15.m128i_i64[0],
    v15.m128i_i64[1],
    v16.m128i_i64[0],
    v16.m128i_i64[1],
    v17.m128i_i64[0],
    v17.m128i_i64[1],
    v18.m128i_i64[0],
    v18.m128i_i64[1],
    v19.m128i_i64[0],
    v19.m128i_i64[1],
    v20.m128i_i64[0],
    v20.m128i_i64[1],
    v21.m128i_i64[0],
    v21.m128i_i64[1],
    v22.m128i_i64[0],
    v22.m128i_i64[1],
    v23.m128i_i64[0],
    v23.m128i_i64[1],
    v24.m128i_i64[0],
    v24.m128i_i64[1],
    v25.m128i_i64[0],
    v25.m128i_i64[1],
    v26.m128i_i64[0],
    v26.m128i_i64[1],
    v27.m128i_i64[0],
    v27.m128i_i64[1],
    v28.m128i_i64[0],
    v28.m128i_i64[1],
    v29.m128i_i64[0],
    v29.m128i_i64[1],
    v30.m128i_i64[0],
    v30.m128i_i64[1],
    v31.m128i_i64[0],
    v31.m128i_i64[1],
    v32.m128i_i64[0],
    v32.m128i_i64[1],
    v33.m128i_i64[0],
    v33.m128i_i64[1],
    v34.m128i_i64[0],
    v34.m128i_i64[1]);
  v10 = *(_BYTE *)(a1 + 16);
  if ( v10 == 3 )
    goto LABEL_31;
LABEL_22:
  if ( v10 == 4 )
    sub_6EE880(a1, 0);
LABEL_12:
  v9 = *(_BYTE *)(a1 + 17);
  if ( v9 == 1 )
  {
    if ( !sub_6ED0A0(a1) )
      goto LABEL_18;
    v9 = *(_BYTE *)(a1 + 17);
  }
  if ( v9 != 3 && (v9 == 2 || sub_6ED0A0(a1) || *(_BYTE *)(a1 + 16) == 5) )
  {
    if ( (_DWORD)a2 )
    {
      v11 = (_QWORD *)sub_6F6F40(a1, 0);
      v12 = (__int64 *)sub_73DC30(116, *v11, v11);
      sub_6E7150(v12, a1);
    }
  }
LABEL_18:
  sub_6E4EE0(a1, (__int64)&v13);
  sub_6E5820(*(unsigned __int64 **)(a1 + 88), 0x2000);
}
