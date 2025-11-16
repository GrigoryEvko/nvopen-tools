// Function: sub_326C5C0
// Address: 0x326c5c0
//
__int64 __fastcall sub_326C5C0(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  const __m128i *v3; // rax
  __int64 v4; // rsi
  __int64 v5; // r8
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 v8; // rcx
  __int64 v9; // r12
  __int64 v10; // r13
  int v11; // edx
  int v12; // eax
  unsigned __int16 *v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v20; // rax
  __int128 v21; // [rsp-30h] [rbp-A0h]
  __int128 v22; // [rsp-10h] [rbp-80h]
  __int128 v23; // [rsp-10h] [rbp-80h]
  __int64 v24; // [rsp-10h] [rbp-80h]
  __int64 v25; // [rsp+8h] [rbp-68h]
  __int64 v26; // [rsp+10h] [rbp-60h]
  __int64 v27; // [rsp+18h] [rbp-58h]
  __int64 v28; // [rsp+18h] [rbp-58h]
  __m128i v29; // [rsp+20h] [rbp-50h]
  __int64 v30; // [rsp+30h] [rbp-40h] BYREF
  int v31; // [rsp+38h] [rbp-38h]

  v2 = a2;
  v3 = *(const __m128i **)(a2 + 40);
  v4 = *(_QWORD *)(a2 + 80);
  v5 = v3->m128i_i64[0];
  v30 = v4;
  v6 = v3[2].m128i_i64[1];
  v7 = v3[3].m128i_i64[0];
  v8 = v6;
  v9 = v3[5].m128i_i64[0];
  v29 = _mm_loadu_si128(v3);
  v10 = v3[5].m128i_i64[1];
  if ( v4 )
  {
    v25 = v2;
    v26 = v3[2].m128i_i64[1];
    v27 = v5;
    sub_B96E90((__int64)&v30, v4, 1);
    v2 = v25;
    v8 = v26;
    v5 = v27;
  }
  v11 = *(_DWORD *)(v8 + 24);
  v31 = *(_DWORD *)(v2 + 72);
  v12 = *(_DWORD *)(v5 + 24);
  if ( v12 != 35 && v12 != 11 || v11 == 35 || v11 == 11 )
  {
    v28 = v2;
    if ( (unsigned __int8)sub_33CF170(v9, v10)
      && ((v13 = *(unsigned __int16 **)(v28 + 48), !*(_BYTE *)(a1 + 33))
       || ((v14 = *v13, v15 = *(_QWORD *)(a1 + 8), v16 = 1, (_WORD)v14 == 1)
        || (_WORD)v14 && (v16 = (unsigned __int16)v14, *(_QWORD *)(v15 + 8 * v14 + 112)))
       && (*(_BYTE *)(v15 + 500 * v16 + 6490) & 0xFB) == 0) )
    {
      *((_QWORD *)&v23 + 1) = v7;
      *(_QWORD *)&v23 = v6;
      v18 = sub_3411F20(
              *(_QWORD *)a1,
              76,
              (unsigned int)&v30,
              (_DWORD)v13,
              *(_DWORD *)(v28 + 68),
              v28,
              *(_OWORD *)&v29,
              v23);
    }
    else
    {
      v17 = sub_326C490((__int64 *)a1, v29.m128i_i64[0], v29.m128i_i64[1], v6, v7, v28, v9, v10);
      if ( v17 )
      {
        v18 = v17;
      }
      else
      {
        v24 = v9;
        v18 = 0;
        v20 = sub_326C490((__int64 *)a1, v6, v7, v29.m128i_i64[0], v29.m128i_i64[1], v28, v24, v10);
        if ( v20 )
          v18 = v20;
      }
    }
  }
  else
  {
    *((_QWORD *)&v22 + 1) = v10;
    *(_QWORD *)&v22 = v9;
    *((_QWORD *)&v21 + 1) = v7;
    *(_QWORD *)&v21 = v6;
    v18 = sub_3412970(
            *(_QWORD *)a1,
            74,
            (unsigned int)&v30,
            *(_QWORD *)(v2 + 48),
            *(_DWORD *)(v2 + 68),
            v2,
            v21,
            *(_OWORD *)&v29,
            v22);
  }
  if ( v30 )
    sub_B91220((__int64)&v30, v30);
  return v18;
}
