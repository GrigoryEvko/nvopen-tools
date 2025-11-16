// Function: sub_2158BD0
// Address: 0x2158bd0
//
void __fastcall sub_2158BD0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  unsigned int v5; // r8d
  __m128i *v6; // rdx
  unsigned int v7; // r8d
  __m128i si128; // xmm0
  _QWORD *v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdi
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  unsigned int v15; // [rsp+Ch] [rbp-1D4h]
  _QWORD v16[2]; // [rsp+10h] [rbp-1D0h] BYREF
  _QWORD *v17; // [rsp+20h] [rbp-1C0h] BYREF
  __int16 v18; // [rsp+30h] [rbp-1B0h]
  _QWORD v19[2]; // [rsp+40h] [rbp-1A0h] BYREF
  __int64 v20; // [rsp+50h] [rbp-190h]
  __m128i *v21; // [rsp+58h] [rbp-188h]
  int v22; // [rsp+60h] [rbp-180h]
  unsigned __int64 *v23; // [rsp+68h] [rbp-178h]
  unsigned __int64 v24[2]; // [rsp+70h] [rbp-170h] BYREF
  _BYTE v25[128]; // [rsp+80h] [rbp-160h] BYREF
  int v26; // [rsp+100h] [rbp-E0h] BYREF
  __int64 v27; // [rsp+108h] [rbp-D8h]
  _BYTE *v28; // [rsp+110h] [rbp-D0h]
  __int64 v29; // [rsp+118h] [rbp-C8h]
  _BYTE v30[128]; // [rsp+120h] [rbp-C0h] BYREF
  int v31; // [rsp+1A0h] [rbp-40h]

  v29 = 0x800000000LL;
  v26 = 0;
  v27 = 0;
  v28 = v30;
  v31 = 0;
  sub_21589E0(a1, a2, (__int64)&v26);
  v3 = *(_QWORD *)(a2 + 56);
  v4 = v3 + 8LL * *(unsigned __int8 *)(a2 + 49);
  if ( v3 != v4 )
  {
    while ( 1 )
    {
      if ( (*(_BYTE *)(*(_QWORD *)v3 + 32LL) & 1) != 0 )
      {
        v5 = *(_DWORD *)(*(_QWORD *)v3 + 72LL);
        if ( v5 != 0x7FFFFFFF )
          break;
      }
      v3 += 8;
      if ( v4 == v3 )
        goto LABEL_5;
    }
    v24[1] = 0x8000000000LL;
    v15 = v5;
    v19[0] = &unk_49EFC48;
    v23 = v24;
    v24[0] = (unsigned __int64)v25;
    v22 = 1;
    v21 = 0;
    v20 = 0;
    v19[1] = 0;
    sub_16E7A40((__int64)v19, 0, 0, 0);
    v6 = v21;
    v7 = v15;
    if ( (unsigned __int64)(v20 - (_QWORD)v21) <= 0x19 )
    {
      v14 = sub_16E7EE0((__int64)v19, "\t.pragma \"used_bytes_mask ", 0x1Au);
      v7 = v15;
      v9 = (_QWORD *)v14;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_4327160);
      v9 = v19;
      v21[1].m128i_i64[0] = 0x73616D5F73657479LL;
      v6[1].m128i_i16[4] = 8299;
      *v6 = si128;
      v21 = (__m128i *)((char *)v21 + 26);
    }
    v10 = sub_16E7A90((__int64)v9, v7);
    v11 = *(_QWORD *)(v10 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v10 + 16) - v11) <= 2 )
    {
      sub_16E7EE0(v10, "\";\n", 3u);
    }
    else
    {
      *(_BYTE *)(v11 + 2) = 10;
      *(_WORD *)v11 = 15138;
      *(_QWORD *)(v10 + 24) += 3LL;
    }
    v12 = *(_QWORD *)(a1 + 256);
    v13 = *v23;
    v16[1] = *((unsigned int *)v23 + 2);
    v18 = 261;
    v16[0] = v13;
    v17 = v16;
    sub_38DD5A0(v12, &v17);
    v19[0] = &unk_49EFD28;
    sub_16E7960((__int64)v19);
    if ( (_BYTE *)v24[0] != v25 )
      _libc_free(v24[0]);
  }
LABEL_5:
  sub_396E900(a1, *(_QWORD *)(a1 + 256), &v26);
  if ( v28 != v30 )
    _libc_free((unsigned __int64)v28);
}
