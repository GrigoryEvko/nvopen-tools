// Function: sub_6F7270
// Address: 0x6f7270
//
__int64 __fastcall sub_6F7270(
        const __m128i *a1,
        __int64 a2,
        _DWORD *a3,
        _BOOL8 a4,
        __int64 a5,
        __int64 a6,
        char a7,
        int a8)
{
  int v9; // r14d
  _BOOL4 v10; // r13d
  _DWORD *v11; // r12
  __int64 v12; // rax
  __int64 v13; // rdi
  int v14; // ebx
  __int8 v15; // al
  int v16; // eax
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rcx
  __int64 v20; // rax
  char i; // dl
  __int64 v23; // rdx
  int v24; // eax
  __int64 **v25; // rax
  int v26; // [rsp+4h] [rbp-1BCh]
  int v27; // [rsp+1Ch] [rbp-1A4h] BYREF
  __int64 *v28; // [rsp+20h] [rbp-1A0h] BYREF
  __int64 v29; // [rsp+28h] [rbp-198h] BYREF
  _OWORD v30[4]; // [rsp+30h] [rbp-190h] BYREF
  _OWORD v31[5]; // [rsp+70h] [rbp-150h] BYREF
  __m128i v32; // [rsp+C0h] [rbp-100h]
  __m128i v33; // [rsp+D0h] [rbp-F0h]
  __m128i v34; // [rsp+E0h] [rbp-E0h]
  __m128i v35; // [rsp+F0h] [rbp-D0h]
  __m128i v36; // [rsp+100h] [rbp-C0h]
  __m128i v37; // [rsp+110h] [rbp-B0h]
  __m128i v38; // [rsp+120h] [rbp-A0h]
  __m128i v39; // [rsp+130h] [rbp-90h]
  __m128i v40; // [rsp+140h] [rbp-80h]
  __m128i v41; // [rsp+150h] [rbp-70h]
  __m128i v42; // [rsp+160h] [rbp-60h]
  __m128i v43; // [rsp+170h] [rbp-50h]
  __m128i v44; // [rsp+180h] [rbp-40h]

  v9 = a6;
  v10 = a4;
  v11 = a3;
  v26 = a5;
  v12 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v13 = a1->m128i_i64[0];
  v29 = v12;
  v14 = sub_8D2EF0(v13);
  v30[0] = _mm_loadu_si128(a1);
  v30[1] = _mm_loadu_si128(a1 + 1);
  v15 = a1[1].m128i_i8[0];
  v30[2] = _mm_loadu_si128(a1 + 2);
  v30[3] = _mm_loadu_si128(a1 + 3);
  v31[0] = _mm_loadu_si128(a1 + 4);
  v31[1] = _mm_loadu_si128(a1 + 5);
  v31[2] = _mm_loadu_si128(a1 + 6);
  v31[3] = _mm_loadu_si128(a1 + 7);
  v31[4] = _mm_loadu_si128(a1 + 8);
  if ( v15 == 2 )
  {
    v32 = _mm_loadu_si128(a1 + 9);
    v33 = _mm_loadu_si128(a1 + 10);
    v34 = _mm_loadu_si128(a1 + 11);
    v35 = _mm_loadu_si128(a1 + 12);
    v36 = _mm_loadu_si128(a1 + 13);
    v37 = _mm_loadu_si128(a1 + 14);
    v38 = _mm_loadu_si128(a1 + 15);
    v39 = _mm_loadu_si128(a1 + 16);
    v40 = _mm_loadu_si128(a1 + 17);
    v41 = _mm_loadu_si128(a1 + 18);
    v42 = _mm_loadu_si128(a1 + 19);
    v43 = _mm_loadu_si128(a1 + 20);
    v44 = _mm_loadu_si128(a1 + 21);
  }
  else if ( v15 == 5 || v15 == 1 )
  {
    v32.m128i_i64[0] = a1[9].m128i_i64[0];
  }
  v16 = sub_6E6010();
  if ( (!dword_4F077BC || !qword_4D03C50 || (*(_BYTE *)(qword_4D03C50 + 18LL) & 0x40) == 0) && !v16 )
    v10 = 0;
  if ( !v11 )
  {
    v11 = (_DWORD *)a1->m128i_i64[0];
    if ( v14 )
      v11 = (_DWORD *)sub_8D46C0(a1->m128i_i64[0]);
  }
  v19 = a1[1].m128i_u8[0];
  if ( !(_BYTE)v19 )
    goto LABEL_14;
  v20 = a1->m128i_i64[0];
  for ( i = *(_BYTE *)(a1->m128i_i64[0] + 140); i == 12; i = *(_BYTE *)(v20 + 140) )
    v20 = *(_QWORD *)(v20 + 160);
  if ( i )
  {
    v27 = 1;
    v23 = qword_4D03C50;
    v24 = *(_DWORD *)(qword_4D03C50 + 16LL);
    if ( (v24 & 0x80100) == 0x80100 && (_BYTE)v19 == 2 && a1[19].m128i_i8[13] != 10 )
    {
      LODWORD(v28) = 0;
      v25 = 0;
      if ( *(char *)(qword_4D03C50 + 18LL) < 0 )
        v25 = &v28;
      sub_710650(
        (_DWORD)a1 + 144,
        a2,
        (_DWORD)v11,
        v29,
        v10,
        1,
        v9,
        a8,
        0,
        (__int64)&v27,
        (__int64)v31 + 4,
        (__int64)v25);
      v23 = (unsigned int)v28;
      if ( (_DWORD)v28 )
        sub_6E50A0();
      if ( !v27 )
      {
        sub_6E6A50(v29, (__int64)a1);
        goto LABEL_15;
      }
      v24 = *(_DWORD *)(qword_4D03C50 + 16LL);
    }
    if ( (v24 & 0x40000100) == 0x40000100
      && (a1[1].m128i_i8[1] != 1
       || sub_6ED0A0((__int64)a1)
       || (unsigned __int8)(*(_BYTE *)(qword_4D03C50 + 16LL) - 2) > 1u) )
    {
      sub_6E68E0(0x1Cu, (__int64)a1);
    }
    else
    {
      v28 = (__int64 *)sub_6F6F40(a1, 0, v23, v19, v17, v18);
      sub_6E7420(a2, v11, v10, 1, v26, v9, a7, (__int64 *)&v28, (_DWORD *)v31 + 1, 0);
      sub_6E7170(v28, (__int64)a1);
    }
  }
  else
  {
LABEL_14:
    sub_6E6870((__int64)a1);
  }
LABEL_15:
  sub_6E4F10((__int64)a1, (__int64)v30, v9, 1);
  return sub_724E30(&v29);
}
