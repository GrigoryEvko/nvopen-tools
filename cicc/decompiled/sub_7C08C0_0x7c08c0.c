// Function: sub_7C08C0
// Address: 0x7c08c0
//
__int64 __fastcall sub_7C08C0(__int64 a1, unsigned int *a2, unsigned int a3, unsigned int a4, int a5, int *a6)
{
  char v9; // al
  unsigned int v10; // r8d
  unsigned __int16 v11; // r9
  int v12; // edi
  __int64 v13; // rsi
  _QWORD *v14; // r13
  _QWORD *v16; // rax
  __int64 v17; // rax
  __m128i *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __m128i v21; // xmm1
  __m128i v22; // xmm2
  __m128i v23; // xmm3
  _BYTE *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  unsigned int v30; // r8d
  unsigned __int16 v31; // r9
  _BYTE *v32; // rdi
  unsigned int v33; // eax
  __m128i v34; // xmm4
  __m128i v35; // xmm5
  __m128i v36; // xmm6
  __m128i v37; // xmm7
  __m128i v38; // xmm4
  __m128i v39; // xmm5
  __m128i v40; // xmm6
  __m128i v41; // xmm7
  unsigned int v42; // [rsp+18h] [rbp-88h]
  unsigned __int16 v43; // [rsp+18h] [rbp-88h]
  unsigned int v44; // [rsp+1Ch] [rbp-84h]
  unsigned int v45; // [rsp+1Ch] [rbp-84h]
  int v46; // [rsp+28h] [rbp-78h] BYREF
  int v47; // [rsp+2Ch] [rbp-74h] BYREF
  __m128i v48; // [rsp+30h] [rbp-70h] BYREF
  __m128i v49; // [rsp+40h] [rbp-60h] BYREF
  __m128i v50; // [rsp+50h] [rbp-50h] BYREF
  __m128i v51[4]; // [rsp+60h] [rbp-40h] BYREF

  if ( !a1 )
  {
    if ( unk_4D04878
      && a5
      && (!dword_4F04D80 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) & 4) == 0)
      && (a4 & 0x4000) != 0 )
    {
LABEL_7:
      v47 = 0;
      v10 = dword_4F063F8;
      v11 = word_4F063FC[0];
      if ( (_WORD)a2 == 43 )
      {
        v42 = word_4F063FC[0];
        v44 = dword_4F063F8;
        v21 = _mm_loadu_si128((const __m128i *)&word_4D04A10);
        v22 = _mm_loadu_si128(&xmmword_4D04A20);
        v23 = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
        v48 = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
        v49 = v21;
        v50 = v22;
        v51[0] = v23;
        sub_7296C0(&v46);
        v24 = (_BYTE *)qword_4F061C8;
        v25 = dword_4F07770;
        ++*(_BYTE *)(qword_4F061C8 + 52LL);
        if ( (_DWORD)v25 )
          ++v24[50];
        ++v24[81];
        ++v24[83];
        sub_7B8B50((unsigned __int64)&v46, a2, v25, (__int64)&dword_4F07770, v44, v42);
        sub_7B8B50((unsigned __int64)&v46, a2, v26, v27, v28, v29);
        ++*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 632);
        v13 = sub_7BF3A0(0, &v47);
        v30 = v44;
        v31 = v42;
        *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
        if ( word_4F06418[0] != 44 )
        {
          sub_7BC0A0(&v47);
          v31 = v42;
          v30 = v44;
        }
        v43 = v31;
        v45 = v30;
        --*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 632);
        sub_729730(v46);
        v10 = v45;
        v11 = v43;
        v32 = (_BYTE *)qword_4F061C8;
        v33 = dword_4F07770;
        --*(_BYTE *)(qword_4F061C8 + 52LL);
        if ( v33 )
          --v32[50];
        --v32[81];
        --v32[83];
        if ( word_4F06418[0] == 44 || (sub_7BEC40(), v10 = v45, v11 = v43, a1) )
        {
          v34 = _mm_loadu_si128(&v48);
          v35 = _mm_loadu_si128(&v49);
          v36 = _mm_loadu_si128(&v50);
          v37 = _mm_loadu_si128(v51);
          word_4F06418[0] = 1;
          *(__m128i *)&qword_4D04A00 = v34;
          v12 = v47;
          *(__m128i *)&word_4D04A10 = v35;
          xmmword_4D04A20 = v36;
          unk_4D04A30 = v37;
        }
        else
        {
          v38 = _mm_loadu_si128(&v48);
          v39 = _mm_loadu_si128(&v49);
          v40 = _mm_loadu_si128(&v50);
          v41 = _mm_loadu_si128(v51);
          word_4F06418[0] = 1;
          v12 = v47;
          *(__m128i *)&qword_4D04A00 = v38;
          *(__m128i *)&word_4D04A10 = v39;
          xmmword_4D04A20 = v40;
          unk_4D04A30 = v41;
          if ( unk_4D04878 )
            goto LABEL_10;
        }
      }
      else
      {
        v12 = 0;
        v13 = 0;
      }
      unk_4D04A12 |= 1u;
      xmmword_4D04A20.m128i_i64[1] = v13;
LABEL_10:
      dword_4F07508[0] = v10;
      LOWORD(dword_4F07508[1]) = v11;
      *a6 = (*a6 | v12) != 0;
      return a1;
    }
    goto LABEL_13;
  }
  v9 = *(_BYTE *)(a1 + 80);
  if ( v9 != 21 )
  {
    if ( unk_4D04878
      && a5
      && (a4 & 0x4000) != 0
      && ((unsigned __int8)(v9 - 10) <= 1u || v9 == 17)
      && (*(_BYTE *)(a1 + 81) & 0x10) == 0 )
    {
      goto LABEL_7;
    }
    if ( v9 != 19 )
    {
      if ( v9 == 3 )
      {
        if ( *(_BYTE *)(a1 + 104) )
        {
          v19 = *(_QWORD *)(a1 + 88);
          if ( ((*(_BYTE *)(v19 + 177) & 0x10) == 0 || !*(_QWORD *)(*(_QWORD *)(v19 + 168) + 168LL))
            && (*(_BYTE *)(v19 + 177) & 0x10) != 0
            && *(_QWORD *)(*(_QWORD *)(v19 + 168) + 168LL) )
          {
            goto LABEL_7;
          }
        }
      }
      else
      {
        if ( (unsigned __int8)(v9 - 20) <= 1u )
          goto LABEL_7;
        if ( ((v9 - 7) & 0xFD) == 0 )
        {
          v20 = *(_QWORD *)(a1 + 88);
          if ( !v20 )
            goto LABEL_13;
          if ( (*(_BYTE *)(v20 + 170) & 0x10) != 0 && **(_QWORD **)(v20 + 216) )
            goto LABEL_7;
        }
        if ( v9 == 17 )
        {
          a2 = (unsigned int *)(unsigned int)a2;
          if ( (unsigned int)sub_8780F0(a1) )
            goto LABEL_7;
        }
      }
    }
LABEL_13:
    v16 = sub_7BF840(a1, a4, a6);
    v14 = v16;
    if ( v16
      && (unsigned __int8)(*((_BYTE *)v16 + 80) - 4) <= 1u
      && (*(_BYTE *)(v16[11] + 177LL) & 0x30) == 0x30
      && (_QWORD *)a1 != v16
      && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v16[12] + 72LL) + 88LL) + 160LL) & 2) != 0
      && (a4 & 0x800) == 0
      && (unsigned __int16)sub_7BE840(0, 0) != 146
      && (*(_BYTE *)(*(_QWORD *)(a1 + 88) + 160LL) & 2) != 0
      && ((dword_4F04C64 == -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) & 2) == 0)
       && (!dword_4F077BC || qword_4F077A8 > 0x9CA4u)
       || (a4 & 0x4000) != 0)
      && (a4 & 0x40000) == 0 )
    {
      v17 = v14[11];
      v14 = (_QWORD *)a1;
      v18 = sub_72F240(*(const __m128i **)(*(_QWORD *)(v17 + 168) + 168LL));
      unk_4D04A12 |= 4u;
      xmmword_4D04A20.m128i_i64[1] = (__int64)v18;
      qword_4D04A18 = (_QWORD *)a1;
    }
    return (__int64)v14;
  }
  return sub_7C7F70(a1, a3, a4, (unsigned __int16)a2, a6);
}
