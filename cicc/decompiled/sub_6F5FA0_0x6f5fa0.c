// Function: sub_6F5FA0
// Address: 0x6f5fa0
//
void __fastcall sub_6F5FA0(const __m128i *a1, _QWORD *a2, int a3, unsigned int a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // rcx
  __int64 v11; // rdx
  char v12; // al
  __int64 v13; // rax
  char i; // dl
  __int64 v15; // rax
  char v16; // dl
  __int64 *v17; // rax
  _DWORD *v18; // rax
  _DWORD *v19; // r14
  char v20; // cl
  char v21; // al
  char v22; // cl
  _OWORD v23[9]; // [rsp+0h] [rbp-180h] BYREF
  __m128i v24; // [rsp+90h] [rbp-F0h]
  __m128i v25; // [rsp+A0h] [rbp-E0h]
  __m128i v26; // [rsp+B0h] [rbp-D0h]
  __m128i v27; // [rsp+C0h] [rbp-C0h]
  __m128i v28; // [rsp+D0h] [rbp-B0h]
  __m128i v29; // [rsp+E0h] [rbp-A0h]
  __m128i v30; // [rsp+F0h] [rbp-90h]
  __m128i v31; // [rsp+100h] [rbp-80h]
  __m128i v32; // [rsp+110h] [rbp-70h]
  __m128i v33; // [rsp+120h] [rbp-60h]
  __m128i v34; // [rsp+130h] [rbp-50h]
  __m128i v35; // [rsp+140h] [rbp-40h]
  __m128i v36; // [rsp+150h] [rbp-30h]

  v8 = (__int64)a1;
  v9 = a1[1].m128i_u8[0];
  v23[0] = _mm_loadu_si128(a1);
  v23[1] = _mm_loadu_si128(a1 + 1);
  v23[2] = _mm_loadu_si128(a1 + 2);
  v23[3] = _mm_loadu_si128(a1 + 3);
  v23[4] = _mm_loadu_si128(a1 + 4);
  v23[5] = _mm_loadu_si128(a1 + 5);
  v23[6] = _mm_loadu_si128(a1 + 6);
  v23[7] = _mm_loadu_si128(a1 + 7);
  v23[8] = _mm_loadu_si128(a1 + 8);
  if ( (_BYTE)v9 == 2 )
  {
    v24 = _mm_loadu_si128(a1 + 9);
    v25 = _mm_loadu_si128(a1 + 10);
    v26 = _mm_loadu_si128(a1 + 11);
    v27 = _mm_loadu_si128(a1 + 12);
    v28 = _mm_loadu_si128(a1 + 13);
    v29 = _mm_loadu_si128(a1 + 14);
    v30 = _mm_loadu_si128(a1 + 15);
    v31 = _mm_loadu_si128(a1 + 16);
    v32 = _mm_loadu_si128(a1 + 17);
    v33 = _mm_loadu_si128(a1 + 18);
    v34 = _mm_loadu_si128(a1 + 19);
    v35 = _mm_loadu_si128(a1 + 20);
    v36 = _mm_loadu_si128(a1 + 21);
    goto LABEL_12;
  }
  if ( (_BYTE)v9 == 5 || (_BYTE)v9 == 1 )
  {
    v24.m128i_i64[0] = a1[9].m128i_i64[0];
    goto LABEL_12;
  }
  if ( (unsigned __int8)(v9 - 3) > 1u )
    goto LABEL_33;
  v11 = a1[8].m128i_i64[1];
  a1 = (const __m128i *)(unsigned int)v9;
  v12 = *(_BYTE *)(v11 + 80);
  if ( v12 == 16 )
  {
    v11 = **(_QWORD **)(v11 + 88);
    v12 = *(_BYTE *)(v11 + 80);
  }
  if ( v12 == 24 )
  {
    v11 = *(_QWORD *)(v11 + 88);
    v12 = *(_BYTE *)(v11 + 80);
  }
  if ( v12 == 17 )
  {
    v11 = *(_QWORD *)(v11 + 88);
    v12 = *(_BYTE *)(v11 + 80);
  }
  if ( v12 != 10 )
    goto LABEL_12;
  v15 = *(_QWORD *)(v11 + 88);
  v16 = *(_BYTE *)(v15 + 174);
  if ( qword_4F04C50 )
  {
    if ( (*(_BYTE *)(v15 + 193) & 0x10) != 0
      || (a6 = 0x8000000000000LL, a5 = *(_QWORD *)(v15 + 200) & 0x8000001000000LL, a5 == 0x8000000000000LL)
      && (*(_BYTE *)(v15 + 192) & 2) == 0 )
    {
      if ( !v16 )
      {
LABEL_32:
        v9 = (unsigned int)a1;
        goto LABEL_33;
      }
      if ( (*(_BYTE *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 198LL) & 0x10) != 0 )
        *(_BYTE *)(v15 + 198) |= 0x18u;
    }
  }
  if ( (a3 || v16 != 1) && v16 != 2 )
  {
    a1 = (const __m128i *)*(unsigned __int8 *)(v8 + 16);
    goto LABEL_32;
  }
  v18 = (_DWORD *)(v8 + 68);
  if ( a2 )
    v18 = a2;
  v19 = v18;
  if ( (unsigned int)sub_6E5430() )
    sub_6851C0(0x1A8u, v19);
  a1 = (const __m128i *)v8;
  sub_6E6840(v8);
  v9 = *(unsigned __int8 *)(v8 + 16);
LABEL_33:
  if ( !(_BYTE)v9 )
  {
LABEL_34:
    sub_6E6870(v8);
    goto LABEL_35;
  }
LABEL_12:
  v13 = *(_QWORD *)v8;
  for ( i = *(_BYTE *)(*(_QWORD *)v8 + 140LL); i == 12; i = *(_BYTE *)(v13 + 140) )
    v13 = *(_QWORD *)(v13 + 160);
  if ( !i )
    goto LABEL_34;
  if ( (_BYTE)v9 == 3 )
  {
    *(_BYTE *)(v8 + 17) = 2;
    sub_6E4EE0(v8, (__int64)v23);
    if ( a2 )
    {
      v20 = *(_BYTE *)(v8 + 19);
      v21 = v20 | 6;
      v22 = v20 | 2;
      *(_BYTE *)(v8 + 19) = v22;
      *(_QWORD *)(v8 + 120) = *a2;
      if ( (*(_BYTE *)(v8 + 18) & 0x28) != 8 )
        v21 = v22;
      *(_BYTE *)(v8 + 19) = v21;
      goto LABEL_37;
    }
LABEL_44:
    sub_6E5010((_BYTE *)v8, v23);
    goto LABEL_38;
  }
  if ( (*(_BYTE *)(v8 + 18) & 1) == 0 && (_BYTE)v9 != 1 )
  {
    if ( (_BYTE)v9 != 2 )
    {
      if ( (_BYTE)v9 == 4 )
      {
        sub_6EE880(v8, a2);
        goto LABEL_35;
      }
LABEL_55:
      sub_721090(a1);
    }
    if ( *(_BYTE *)(v8 + 317) != 12 || *(_BYTE *)(v8 + 320) != 1 )
      goto LABEL_55;
    v17 = (__int64 *)sub_72E9A0(v8 + 144);
    sub_6E70E0(v17, v8);
    *(_BYTE *)(v8 + 17) = 3;
  }
  sub_6F5960((__m128i *)v8, a4, a2, v9, a5, a6);
LABEL_35:
  sub_6E4EE0(v8, (__int64)v23);
  if ( !a2 )
    goto LABEL_44;
  *(_QWORD *)(v8 + 68) = *a2;
  sub_6E3280(v8, a2);
LABEL_37:
  *(_BYTE *)(v8 + 18) &= ~8u;
LABEL_38:
  sub_6E5820(*(unsigned __int64 **)(v8 + 88), 32);
}
