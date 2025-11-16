// Function: sub_87A120
// Address: 0x87a120
//
__int64 __fastcall sub_87A120(unsigned __int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int *v6; // r12
  unsigned __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned int *v12; // rsi
  unsigned __int64 v13; // rdi
  int v14; // r14d
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned __int16 v19; // ax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r12
  int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rdx
  __int64 v30; // rcx
  __m128i v31; // xmm1
  __m128i v32; // xmm2
  __m128i v33; // xmm3
  unsigned int *v34; // [rsp-10h] [rbp-A0h]
  unsigned __int64 v35; // [rsp-8h] [rbp-98h]
  unsigned int v36; // [rsp+Ch] [rbp-84h] BYREF
  __int64 v37; // [rsp+10h] [rbp-80h] BYREF
  unsigned __int64 v38; // [rsp+18h] [rbp-78h] BYREF
  __int64 v39[2]; // [rsp+20h] [rbp-70h] BYREF
  __m128i v40; // [rsp+30h] [rbp-60h]
  __m128i v41; // [rsp+40h] [rbp-50h]
  __m128i v42; // [rsp+50h] [rbp-40h]

  v6 = a2;
  if ( a1 )
  {
    v7 = a1;
    a2 = (unsigned int *)qword_4F06410;
    a1 = dword_4F073B8[0];
    *(_QWORD *)(v7 + 80) = sub_724840(dword_4F073B8[0], qword_4F06410);
  }
  sub_7B8B50(a1, a2, a3, a4, a5, a6);
  if ( word_4F06418[0] == 27 )
  {
    sub_7B8B50(a1, a2, v8, v9, v10, v11);
    v14 = 0;
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    if ( word_4F06418[0] != 7 )
    {
      v12 = dword_4F07508;
      v13 = 1038;
      sub_684B30(0x40Eu, dword_4F07508);
      v19 = word_4F06418[0];
      if ( word_4F06418[0] == 28 )
        goto LABEL_6;
      goto LABEL_13;
    }
  }
  else
  {
    v12 = dword_4F07508;
    v13 = 125;
    v14 = 1;
    sub_684B30(0x7Du, dword_4F07508);
    v17 = qword_4F061C8;
    v18 = *(unsigned __int8 *)(qword_4F061C8 + 36LL);
    *(_BYTE *)(qword_4F061C8 + 36LL) = v18 + 1;
    v19 = word_4F06418[0];
    if ( word_4F06418[0] != 7 )
    {
      if ( word_4F06418[0] == 28 )
      {
LABEL_6:
        sub_7B8B50(v13, dword_4F07508, v17, v18, v15, v16);
        goto LABEL_7;
      }
      goto LABEL_14;
    }
  }
  if ( *qword_4F06410 != 34 )
  {
    v12 = dword_4F07508;
    v13 = 1434;
    sub_684B30(0x59Au, dword_4F07508);
    v19 = word_4F06418[0];
    if ( word_4F06418[0] != 28 )
    {
LABEL_13:
      v17 = qword_4F061C8;
      v18 = (unsigned int)*(unsigned __int8 *)(qword_4F061C8 + 36LL) - 1;
LABEL_14:
      *(_BYTE *)(v17 + 36) = v18;
      if ( v19 == 10 )
        return 0;
LABEL_8:
      if ( (unsigned __int16)(v19 - 9) > 1u )
      {
        do
          sub_7B8B50(v13, v12, v17, v18, v15, v16);
        while ( (unsigned __int16)(word_4F06418[0] - 9) > 1u );
      }
      return 0;
    }
    sub_7B8B50(0x59Au, dword_4F07508, v20, v21, v15, v16);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    if ( word_4F06418[0] == 10 )
      return 0;
LABEL_25:
    v19 = word_4F06418[0];
    goto LABEL_8;
  }
  v37 = 0;
  *(_QWORD *)v6 = *(_QWORD *)&dword_4F063F8;
  v22 = 0;
  qword_4F06460 = qword_4F06410 + 1;
  v24 = sub_7B6B00(&v37, 0, 17, 34, 0, -1, qword_4F06410, 0);
  v12 = v34;
  v13 = v35;
  if ( !v24 )
  {
    sub_7CE2C0((unsigned __int64)(qword_4F06410 + 1), (_BYTE *)qword_4F06408, 17, v37, &v36, &v38, 0);
    ++qword_4F06460;
    if ( v36 )
    {
      sub_7B0EB0(v38, (__int64)v39);
      v13 = v36;
      v12 = (unsigned int *)v39;
      sub_684B30(v36, v39);
    }
    else
    {
      v31 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v32 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v33 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v39[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
      v40 = v31;
      v41 = v32;
      v42 = v33;
      v39[1] = *(_QWORD *)&dword_4F077C8;
      v13 = (unsigned __int64)qword_4F063B8;
      v12 = (unsigned int *)(*(_QWORD *)word_4F063B0 - 1LL);
      v22 = sub_87A100(qword_4F063B8, *(_QWORD *)word_4F063B0 - 1LL, v39);
    }
  }
  sub_7B8B50(v13, v12, v25, v26, v27, v28);
  v19 = word_4F06418[0];
  if ( word_4F06418[0] == 28 )
  {
    sub_7B8B50(v13, v12, v29, v30, v15, v16);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    if ( word_4F06418[0] == 10 )
    {
      if ( !v14 )
        return v22;
      return 0;
    }
    if ( !v14 )
      return v22;
    goto LABEL_25;
  }
  if ( v14 )
    goto LABEL_13;
  v12 = dword_4F07508;
  v13 = 18;
  sub_684B30(0x12u, dword_4F07508);
LABEL_7:
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  v19 = word_4F06418[0];
  if ( word_4F06418[0] != 10 )
    goto LABEL_8;
  return 0;
}
