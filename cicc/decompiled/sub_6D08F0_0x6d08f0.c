// Function: sub_6D08F0
// Address: 0x6d08f0
//
_DWORD *__fastcall sub_6D08F0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // r13
  __int64 v4; // r14
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // r12
  int v12; // eax
  __int64 v13; // rsi
  _DWORD *v14; // r15
  unsigned int v15; // r12d
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rdx
  char v20; // al
  int v21; // esi
  int v22; // [rsp+4h] [rbp-24Ch] BYREF
  __int64 v23; // [rsp+8h] [rbp-248h] BYREF
  __int64 v24; // [rsp+10h] [rbp-240h] BYREF
  _BYTE v25[8]; // [rsp+18h] [rbp-238h] BYREF
  _BYTE v26[160]; // [rsp+20h] [rbp-230h] BYREF
  __m128i v27; // [rsp+C0h] [rbp-190h] BYREF
  __m128i v28; // [rsp+D0h] [rbp-180h]
  __m128i v29; // [rsp+E0h] [rbp-170h]
  __m128i v30; // [rsp+F0h] [rbp-160h]
  __m256i v31; // [rsp+100h] [rbp-150h]
  __m128i v32; // [rsp+120h] [rbp-130h]
  __m128i v33; // [rsp+130h] [rbp-120h]
  __m128i v34; // [rsp+140h] [rbp-110h]
  __m128i v35; // [rsp+150h] [rbp-100h]
  __m128i v36; // [rsp+160h] [rbp-F0h]
  __m128i v37; // [rsp+170h] [rbp-E0h]
  __m128i v38; // [rsp+180h] [rbp-D0h]
  __m128i v39; // [rsp+190h] [rbp-C0h]
  __m128i v40; // [rsp+1A0h] [rbp-B0h]
  __m128i v41; // [rsp+1B0h] [rbp-A0h]
  __m128i v42; // [rsp+1C0h] [rbp-90h]
  __m128i v43; // [rsp+1D0h] [rbp-80h]
  __m128i v44; // [rsp+1E0h] [rbp-70h]
  __m128i v45; // [rsp+1F0h] [rbp-60h]
  __m128i v46; // [rsp+200h] [rbp-50h]
  __m128i v47; // [rsp+210h] [rbp-40h]

  v3 = *(_QWORD *)(a1 + 80);
  sub_6E1DD0(&v23);
  sub_6E1E00(4, v26, 0, 0);
  if ( word_4F06418[0] == 73 && dword_4D04428 )
  {
    v9 = sub_6BA760(0, 0);
    v10 = sub_6E1A20(v9);
    v11 = sub_72B6D0(v10, 0);
    v12 = sub_6E1A20(v9);
    v13 = v11;
    if ( (unsigned int)sub_83D110(v11, v11, 0, 0, v9, v12, (__int64)&v24, (__int64)v25, (__int64)&v22) )
    {
      if ( !v22 )
      {
        v21 = v24;
        goto LABEL_21;
      }
    }
    else if ( !v22 )
    {
      if ( *(_QWORD *)(v3 + 8) )
      {
        v14 = (_DWORD *)sub_6E1A20(v9);
        v15 = (*(_BYTE *)(*(_QWORD *)(v3 + 8) + 175LL) & 2) == 0 ? 3032 : 2544;
        if ( (unsigned int)sub_6E5430(v9, v13, v16, v17, v18) )
          sub_6851C0(v15, v14);
      }
      sub_6E6500(v9);
      v19 = *(_QWORD *)(v9 + 24);
      v27 = _mm_loadu_si128((const __m128i *)(v19 + 8));
      v28 = _mm_loadu_si128((const __m128i *)(v19 + 24));
      v29 = _mm_loadu_si128((const __m128i *)(v19 + 40));
      v30 = _mm_loadu_si128((const __m128i *)(v19 + 56));
      *(__m128i *)v31.m256i_i8 = _mm_loadu_si128((const __m128i *)(v19 + 72));
      *(__m128i *)&v31.m256i_u64[2] = _mm_loadu_si128((const __m128i *)(v19 + 88));
      v32 = _mm_loadu_si128((const __m128i *)(v19 + 104));
      v33 = _mm_loadu_si128((const __m128i *)(v19 + 120));
      v34 = _mm_loadu_si128((const __m128i *)(v19 + 136));
      v20 = *(_BYTE *)(v19 + 24);
      switch ( v20 )
      {
        case 2:
          v35 = _mm_loadu_si128((const __m128i *)(v19 + 152));
          v36 = _mm_loadu_si128((const __m128i *)(v19 + 168));
          v37 = _mm_loadu_si128((const __m128i *)(v19 + 184));
          v38 = _mm_loadu_si128((const __m128i *)(v19 + 200));
          v39 = _mm_loadu_si128((const __m128i *)(v19 + 216));
          v40 = _mm_loadu_si128((const __m128i *)(v19 + 232));
          v41 = _mm_loadu_si128((const __m128i *)(v19 + 248));
          v42 = _mm_loadu_si128((const __m128i *)(v19 + 264));
          v43 = _mm_loadu_si128((const __m128i *)(v19 + 280));
          v44 = _mm_loadu_si128((const __m128i *)(v19 + 296));
          v45 = _mm_loadu_si128((const __m128i *)(v19 + 312));
          v46 = _mm_loadu_si128((const __m128i *)(v19 + 328));
          v47 = _mm_loadu_si128((const __m128i *)(v19 + 344));
          break;
        case 5:
          v35.m128i_i64[0] = *(_QWORD *)(v19 + 152);
          break;
        case 1:
          v35.m128i_i64[0] = *(_QWORD *)(v19 + 152);
          break;
      }
      goto LABEL_22;
    }
    v21 = dword_4D03B80;
    v24 = *(_QWORD *)&dword_4D03B80;
LABEL_21:
    sub_839D30(v9, v21, 0, 1, 0, 0, 1, 1, 0, (__int64)&v27, 0, 0);
LABEL_22:
    sub_6E1990(v9);
    goto LABEL_4;
  }
  sub_69ED20((__int64)&v27, 0, 0, 0);
  sub_6F69D0(&v27, 6);
LABEL_4:
  v4 = v27.m128i_i64[0];
  if ( v28.m128i_i8[1] == 2 || (unsigned int)sub_6ED0A0(&v27) )
  {
    v5 = sub_72D6A0(v4);
  }
  else if ( (unsigned int)sub_8D3D40(v4) )
  {
    v5 = *(_QWORD *)&dword_4D03B80;
  }
  else
  {
    v5 = sub_72D600(v4);
  }
  v6 = sub_736020(v5, 0);
  *(_QWORD *)(v3 + 16) = v6;
  v7 = v6;
  *a2 = *(__int64 *)((char *)v31.m256i_i64 + 4);
  sub_68BC10(v6, &v27);
  sub_6E2B30(v7, &v27);
  sub_6E1DF0(v23);
  *(_QWORD *)&dword_4F061D8 = *(__int64 *)((char *)&v31.m256i_i64[1] + 4);
  return &dword_4F061D8;
}
