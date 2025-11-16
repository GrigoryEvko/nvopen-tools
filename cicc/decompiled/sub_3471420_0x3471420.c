// Function: sub_3471420
// Address: 0x3471420
//
unsigned __int8 *__fastcall sub_3471420(__int64 a1, __int64 a2, _QWORD *a3, __m128i a4)
{
  unsigned __int8 *result; // rax
  unsigned __int16 *v7; // rax
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rsi
  _BOOL4 v11; // ebx
  unsigned int v12; // ebx
  __int64 v13; // rax
  const __m128i *v14; // rax
  int v15; // r9d
  __m128i v16; // xmm0
  unsigned __int8 *v17; // r14
  unsigned __int64 v18; // r15
  int v19; // eax
  unsigned int v20; // esi
  __int64 v21; // rax
  unsigned __int8 *v22; // rcx
  unsigned __int8 *v23; // rax
  unsigned int v24; // edx
  unsigned int v25; // edx
  __int128 v26; // [rsp-10h] [rbp-A0h]
  __int128 v27; // [rsp+10h] [rbp-80h]
  unsigned __int8 *v28; // [rsp+10h] [rbp-80h]
  unsigned int v29; // [rsp+40h] [rbp-50h] BYREF
  __int64 v30; // [rsp+48h] [rbp-48h]
  __int64 v31; // [rsp+50h] [rbp-40h] BYREF
  int v32; // [rsp+58h] [rbp-38h]

  result = sub_3471020(a1, a2, a3, a4);
  if ( !result )
  {
    v7 = *(unsigned __int16 **)(a2 + 48);
    v8 = *v7;
    v9 = *((_QWORD *)v7 + 1);
    LOWORD(v29) = v8;
    v30 = v9;
    if ( (_WORD)v8 )
    {
      if ( (unsigned __int16)(v8 - 176) > 0x34u )
      {
LABEL_5:
        v10 = *(_QWORD *)(a2 + 80);
        v31 = v10;
        if ( v10 )
          sub_B96E90((__int64)&v31, v10, 1);
        v11 = *(_DWORD *)(a2 + 24) != 279;
        v32 = *(_DWORD *)(a2 + 72);
        v12 = v11 + 281;
        v13 = 1;
        if ( (_WORD)v8 == 1
          || (_WORD)v8 && (v13 = (unsigned __int16)v8, *(_QWORD *)(a1 + 8LL * (unsigned __int16)v8 + 112)) )
        {
          if ( (*(_BYTE *)(v12 + a1 + 500 * v13 + 6414) & 0xFB) == 0 )
          {
            v14 = *(const __m128i **)(a2 + 40);
            v15 = *(_DWORD *)(a2 + 28);
            v16 = _mm_loadu_si128(v14);
            v17 = (unsigned __int8 *)v14[2].m128i_i64[1];
            v18 = v14[3].m128i_u64[0];
            v27 = (__int128)v16;
            if ( (v15 & 0x20) == 0 )
            {
              if ( !(unsigned __int8)sub_33CE830((_QWORD **)a3, v16.m128i_i64[0], v16.m128i_i64[1], 1u, 0) )
              {
                *(_QWORD *)&v27 = sub_33FA050(
                                    (__int64)a3,
                                    154,
                                    (__int64)&v31,
                                    v29,
                                    v30,
                                    *(_DWORD *)(a2 + 28),
                                    (unsigned __int8 *)v16.m128i_i64[0],
                                    v16.m128i_i64[1]);
                *((_QWORD *)&v27 + 1) = v25 | v16.m128i_i64[1] & 0xFFFFFFFF00000000LL;
              }
              if ( (unsigned __int8)sub_33CE830((_QWORD **)a3, (__int64)v17, v18, 1u, 0) )
              {
                v15 = *(_DWORD *)(a2 + 28);
              }
              else
              {
                v23 = sub_33FA050((__int64)a3, 154, (__int64)&v31, v29, v30, *(_DWORD *)(a2 + 28), v17, v18);
                v15 = *(_DWORD *)(a2 + 28);
                v17 = v23;
                v18 = v24 | v18 & 0xFFFFFFFF00000000LL;
              }
            }
            *((_QWORD *)&v26 + 1) = v18;
            *(_QWORD *)&v26 = v17;
            result = sub_3405C90(a3, v12, (__int64)&v31, v29, v30, v15, v16, v27, v26);
            goto LABEL_11;
          }
        }
        v19 = *(_DWORD *)(a2 + 28);
        if ( (v19 & 0x20) == 0 )
        {
          if ( !(unsigned __int8)sub_33CE830(
                                   (_QWORD **)a3,
                                   **(_QWORD **)(a2 + 40),
                                   *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
                                   0,
                                   0)
            || !(unsigned __int8)sub_33CE830(
                                   (_QWORD **)a3,
                                   *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                                   *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                                   0,
                                   0) )
          {
            goto LABEL_22;
          }
          v19 = *(_DWORD *)(a2 + 28);
        }
        if ( (v19 & 0x80u) != 0
          || (unsigned __int8)sub_33CEB60(a3, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL))
          || (unsigned __int8)sub_33CEB60(
                                a3,
                                *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                                *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL)) )
        {
          v20 = (*(_DWORD *)(a2 + 24) != 279) + 283;
          if ( (_WORD)v8 == 1 )
          {
            v21 = 1;
          }
          else
          {
            if ( !(_WORD)v8 )
              goto LABEL_22;
            v21 = (unsigned __int16)v8;
            if ( !*(_QWORD *)(a1 + 8 * v8 + 112) )
              goto LABEL_22;
          }
          if ( (*(_BYTE *)(v20 + a1 + 500 * v21 + 6414) & 0xFB) == 0 )
          {
            result = sub_3405C90(
                       a3,
                       v20,
                       (__int64)&v31,
                       v29,
                       v30,
                       *(_DWORD *)(a2 + 28),
                       a4,
                       *(_OWORD *)*(_QWORD *)(a2 + 40),
                       *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
LABEL_11:
            if ( v31 )
            {
              v28 = result;
              sub_B91220((__int64)&v31, v31);
              return v28;
            }
            return result;
          }
        }
LABEL_22:
        v22 = sub_3452270(a1, a2, a3);
        result = 0;
        if ( v22 )
          result = v22;
        goto LABEL_11;
      }
    }
    else if ( !sub_3007100((__int64)&v29) )
    {
      goto LABEL_5;
    }
    sub_C64ED0("Expanding fminnum/fmaxnum for scalable vectors is undefined.", 1u);
  }
  return result;
}
