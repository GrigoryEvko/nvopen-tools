// Function: sub_7C7F70
// Address: 0x7c7f70
//
_QWORD *__fastcall sub_7C7F70(__int64 a1, int a2, unsigned int a3, __int16 a4, int *a5)
{
  _QWORD *v5; // r13
  int v8; // edi
  char v9; // si
  __m128i v11; // xmm1
  __m128i v12; // xmm2
  __m128i v13; // xmm3
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  _BYTE *v18; // rax
  unsigned __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  int v24; // edi
  _BYTE *v25; // rax
  unsigned int v26; // esi
  __int64 v27; // rcx
  __int64 v28; // rax
  __m128i v29; // xmm5
  __m128i v30; // xmm6
  __m128i v31; // xmm7
  __int64 *v32; // rax
  __int64 v33; // rdx
  int v35; // [rsp+10h] [rbp-90h] BYREF
  int v36; // [rsp+14h] [rbp-8Ch] BYREF
  __int64 *i; // [rsp+18h] [rbp-88h] BYREF
  __int64 v38; // [rsp+20h] [rbp-80h] BYREF
  __int64 v39; // [rsp+28h] [rbp-78h] BYREF
  __m128i v40; // [rsp+30h] [rbp-70h] BYREF
  __m128i v41; // [rsp+40h] [rbp-60h] BYREF
  __m128i v42; // [rsp+50h] [rbp-50h] BYREF
  __m128i v43[4]; // [rsp+60h] [rbp-40h] BYREF

  v5 = (_QWORD *)a1;
  i = 0;
  v36 = 0;
  v39 = *(_QWORD *)&dword_4F063F8;
  if ( a4 == 43 )
  {
    v11 = _mm_loadu_si128((const __m128i *)&word_4D04A10);
    v12 = _mm_loadu_si128(&xmmword_4D04A20);
    v13 = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
    v40 = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
    v41 = v11;
    v42 = v12;
    v43[0] = v13;
    sub_7296C0(&v35);
    v18 = (_BYTE *)qword_4F061C8;
    v19 = dword_4F07770;
    ++*(_BYTE *)(qword_4F061C8 + 52LL);
    if ( (_DWORD)v19 )
      ++v18[50];
    ++v18[81];
    ++v18[83];
    sub_7B8B50(v19, &dword_4F07770, v14, v15, v16, v17);
    sub_7B8B50(v19, &dword_4F07770, v20, v21, v22, v23);
    ++*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 632);
    if ( v5 && (*(_BYTE *)(v5[11] + 160LL) & 6) == 0 )
    {
      v38 = -1;
      v32 = sub_7C6D90((__int64)v5, 0, &v36, a3, &v38);
      v33 = v38;
      for ( i = v32; v32; v32 = (__int64 *)*v32 )
      {
        if ( !v33 )
          break;
        --v33;
        *((_BYTE *)v32 + 24) |= 2u;
        v38 = v33;
      }
    }
    else
    {
      i = (__int64 *)sub_7BF3A0(1u, &v36);
    }
    *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
    if ( word_4F06418[0] != 44 )
      sub_7BC0A0(&v36);
    v24 = v35;
    --*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 632);
    sub_729730(v24);
    v25 = (_BYTE *)qword_4F061C8;
    v26 = dword_4F07770;
    --*(_BYTE *)(qword_4F061C8 + 52LL);
    if ( v26 )
      --v25[50];
    --v25[81];
    --v25[83];
    if ( word_4F06418[0] != 44 )
      sub_7BEC40();
    if ( !i || v36 )
    {
      sub_885B10(&qword_4D04A00);
      v5 = qword_4D04A18;
      v8 = v36;
      qword_4D04A08 = v40.m128i_i64[1];
    }
    else
    {
      if ( *(_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 768) == a2 )
      {
        v5 = 0;
        sub_725130(i);
        v8 = v36;
        word_4F06418[0] = 1;
        goto LABEL_6;
      }
      v27 = a3 & 0x4000;
      if ( (a3 & 0x4000) != 0 )
      {
        if ( !qword_4D03C50 || (v27 = 1, (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) == 0) )
        {
          v27 = 0;
          v28 = *(_QWORD *)(*(_QWORD *)(v5[11] + 104LL) + 192LL);
          if ( v28 )
          {
            v27 = 1;
            if ( (*(_BYTE *)(v28 + 175) & 7) == 0 )
              v27 = (unsigned int)sub_8D23E0(*(_QWORD *)(v28 + 120)) != 0;
          }
        }
      }
      v5 = (_QWORD *)sub_8C0230(v5, &i, (a3 & 0x400) != 0, v27, 1);
      if ( v5 )
      {
        v8 = v36;
      }
      else
      {
        v36 = 1;
        v8 = 1;
      }
      v29 = _mm_loadu_si128(&v41);
      v30 = _mm_loadu_si128(&v42);
      v31 = _mm_loadu_si128(v43);
      *(__m128i *)&qword_4D04A00 = _mm_loadu_si128(&v40);
      *(__m128i *)&word_4D04A10 = v29;
      xmmword_4D04A20 = v30;
      unk_4D04A30 = v31;
    }
    v9 = 1;
    word_4F06418[0] = 1;
  }
  else if ( (a3 & 1) != 0 )
  {
    v8 = 0;
    v9 = 0;
  }
  else
  {
    sub_6854C0(0x1B9u, (FILE *)&v39, a1);
    sub_885B10(&qword_4D04A00);
    v5 = qword_4D04A18;
    v8 = 1;
    v36 = 1;
    v9 = 1;
  }
  if ( v5 )
  {
    qword_4D04A18 = v5;
    HIBYTE(word_4D04A10) = (v9 << 6) | HIBYTE(word_4D04A10) & 0xBF;
    qword_4D04A00 = *v5;
    unk_4D04A12 = v9 | unk_4D04A12 & 0xFE;
    xmmword_4D04A20.m128i_i64[1] = (__int64)i;
  }
LABEL_6:
  *(_QWORD *)dword_4F07508 = v39;
  *a5 = v8;
  return v5;
}
