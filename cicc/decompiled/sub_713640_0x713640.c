// Function: sub_713640
// Address: 0x713640
//
__int64 __fastcall sub_713640(
        const __m128i *a1,
        const __m128i *a2,
        _QWORD *a3,
        unsigned int *a4,
        _DWORD *a5,
        unsigned __int8 *a6)
{
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __m128i *v15; // rax
  __int64 v16; // rcx
  __int64 result; // rax
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int8 v22; // al
  __m128i *v23; // rdi
  int v24; // eax
  __int8 v25; // al
  int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 i; // r15
  unsigned __int64 v33; // rsi
  int v34; // eax
  int v35; // [rsp+10h] [rbp-80h]
  _BOOL4 v37; // [rsp+28h] [rbp-68h] BYREF
  unsigned int v38; // [rsp+2Ch] [rbp-64h] BYREF
  __m128i *v39; // [rsp+30h] [rbp-60h] BYREF
  __m128i *v40; // [rsp+38h] [rbp-58h] BYREF
  __m128i v41; // [rsp+40h] [rbp-50h] BYREF
  __m128i v42[4]; // [rsp+50h] [rbp-40h] BYREF

  v39 = (__m128i *)sub_724DC0(a1, a2, a3, a4, a5, a6);
  v15 = (__m128i *)sub_724DC0(a1, a2, v11, v12, v13, v14);
  *a5 = 0;
  *a6 = 5;
  v40 = v15;
  if ( sub_70E2C0((__int64)a1, (__int64)a2, &v38, v16) )
  {
    v22 = a1[10].m128i_i8[13];
    v23 = v40;
    if ( v22 == 1 )
    {
      *v40 = _mm_loadu_si128(a1);
      v23[1] = _mm_loadu_si128(a1 + 1);
      v23[2] = _mm_loadu_si128(a1 + 2);
      v23[3] = _mm_loadu_si128(a1 + 3);
      v23[4] = _mm_loadu_si128(a1 + 4);
      v23[5] = _mm_loadu_si128(a1 + 5);
      v23[6] = _mm_loadu_si128(a1 + 6);
      v23[7] = _mm_loadu_si128(a1 + 7);
      v23[8] = _mm_loadu_si128(a1 + 8);
      v23[9] = _mm_loadu_si128(a1 + 9);
      v23[10] = _mm_loadu_si128(a1 + 10);
      v23[11] = _mm_loadu_si128(a1 + 11);
      v23[12] = _mm_loadu_si128(a1 + 12);
    }
    else
    {
      if ( v22 != 6 )
        goto LABEL_38;
      sub_72BAF0(v40, a1[12].m128i_i64[0], unk_4F06A60);
    }
    v24 = sub_620E90((__int64)v40);
    v23 = v39;
    v35 = v24;
    v25 = a2[10].m128i_i8[13];
    if ( v25 == 1 )
    {
      *v39 = _mm_loadu_si128(a2);
      v23[1] = _mm_loadu_si128(a2 + 1);
      v23[2] = _mm_loadu_si128(a2 + 2);
      v23[3] = _mm_loadu_si128(a2 + 3);
      v23[4] = _mm_loadu_si128(a2 + 4);
      v23[5] = _mm_loadu_si128(a2 + 5);
      v23[6] = _mm_loadu_si128(a2 + 6);
      v23[7] = _mm_loadu_si128(a2 + 7);
      v23[8] = _mm_loadu_si128(a2 + 8);
      v23[9] = _mm_loadu_si128(a2 + 9);
      v23[10] = _mm_loadu_si128(a2 + 10);
      v23[11] = _mm_loadu_si128(a2 + 11);
      v23[12] = _mm_loadu_si128(a2 + 12);
      goto LABEL_14;
    }
    if ( v25 == 6 )
    {
      sub_72BAF0(v39, a2[12].m128i_i64[0], unk_4F06A60);
LABEL_14:
      v26 = sub_620E90((__int64)v39);
      v41 = _mm_loadu_si128(v40 + 11);
      sub_621670(&v41, v35, v39[11].m128i_i16, v26, &v37);
      if ( v37 )
        goto LABEL_15;
      if ( !(unsigned int)sub_8D2780(a1[8].m128i_i64[0]) )
      {
        for ( i = sub_8D46C0(a1[8].m128i_i64[0]); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        if ( dword_4F077C0 && ((unsigned int)sub_8D2600(i) || (unsigned int)sub_8D2310(i)) )
          v33 = 1;
        else
          v33 = *(_QWORD *)(i + 128);
        sub_620DE0(v42, v33);
        v34 = sub_620E90((__int64)a3);
        sub_6220A0(&v41, v42, v34, &v37);
      }
      if ( v37 )
      {
LABEL_15:
        *a5 = 61;
        *a6 = 8;
      }
      else
      {
        sub_70FF50(&v41, (__int64)a3, 1, 0, a5, a6);
      }
      goto LABEL_3;
    }
LABEL_38:
    sub_721090(v23);
  }
  if ( !v38 )
  {
    v18 = HIDWORD(qword_4F077B4);
    if ( HIDWORD(qword_4F077B4) )
    {
      if ( a1[10].m128i_i8[13] == 6 && a1[11].m128i_i8[0] == 6 && a2[10].m128i_i8[13] == 6 && a2[11].m128i_i8[0] == 6 )
      {
        sub_724C70(a3, 8);
        a3[22] = sub_73A460(a2);
        a3[23] = sub_73A460(a1);
        a3[16] = sub_72BA30(unk_4F06A60);
        goto LABEL_3;
      }
      if ( sub_70FCE0((__int64)a2) )
      {
        if ( (unsigned int)sub_711520((__int64)a2, v18, v19, v20, v21) )
        {
          if ( (unsigned int)sub_8D2E30(a1[8].m128i_i64[0]) )
          {
            v27 = sub_8D46C0(a1[8].m128i_i64[0]);
            if ( (unsigned int)sub_8D29E0(v27) )
            {
              sub_72A510(a1, a3);
              v28 = sub_72BA30(unk_4F06A60);
              sub_70FEE0((__int64)a3, v28, v29, v30, v31);
              goto LABEL_3;
            }
          }
        }
      }
    }
    v38 = 1;
  }
LABEL_3:
  sub_724E30(&v39);
  sub_724E30(&v40);
  result = v38;
  *a4 = v38;
  return result;
}
