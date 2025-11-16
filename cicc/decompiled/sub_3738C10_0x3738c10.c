// Function: sub_3738C10
// Address: 0x3738c10
//
__int64 __fastcall sub_3738C10(__int64 *a1, __int64 a2, __int16 a3, __int64 a4)
{
  __int64 v8; // rdi
  __m128i *v9; // rsi
  _QWORD *v10; // rax
  int v11; // r15d
  __int64 v12; // rax
  __int64 v13; // r15
  _QWORD *v14; // rax
  __int64 *v15; // r8
  __int64 v16; // rdx
  unsigned __int64 **v17; // r13
  __int64 result; // rax
  __int64 v19; // r15
  unsigned __int16 v20; // ax
  unsigned __int64 **v21; // r8
  __int16 v22; // r14
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 **v26; // r15
  unsigned int v27; // r12d
  unsigned int v28; // r13d
  __int64 v29; // [rsp+0h] [rbp-50h]
  __int64 *v30; // [rsp+0h] [rbp-50h]
  _QWORD *v31; // [rsp+8h] [rbp-48h]
  unsigned __int64 **v32; // [rsp+8h] [rbp-48h]
  __m128i v33; // [rsp+10h] [rbp-40h] BYREF

  v8 = a1[26];
  if ( a1[51] )
  {
    if ( !a4 )
      goto LABEL_7;
  }
  else if ( *(_BYTE *)(v8 + 3769) || !a4 )
  {
    goto LABEL_21;
  }
  v33.m128i_i64[0] = a4;
  v33.m128i_i64[1] = (__int64)a1;
  v9 = *(__m128i **)(v8 + 712);
  if ( v9 == *(__m128i **)(v8 + 720) )
  {
    sub_3223B70((unsigned __int64 *)(v8 + 704), v9, &v33);
    v8 = a1[26];
  }
  else
  {
    if ( v9 )
    {
      *v9 = _mm_loadu_si128(&v33);
      v9 = *(__m128i **)(v8 + 712);
    }
    *(_QWORD *)(v8 + 712) = v9 + 1;
    v8 = a1[26];
  }
LABEL_7:
  if ( *(_BYTE *)(v8 + 3769) && a1[51] )
  {
    v10 = *(_QWORD **)a4;
    v11 = *(_DWORD *)(v8 + 3760);
    if ( *(_QWORD *)a4 )
      goto LABEL_10;
LABEL_23:
    if ( (*(_BYTE *)(a4 + 9) & 0x70) != 0x20 )
      goto LABEL_24;
    if ( *(char *)(a4 + 8) < 0 )
      goto LABEL_24;
    *(_BYTE *)(a4 + 8) |= 8u;
    v10 = sub_E807D0(*(_QWORD *)(a4 + 24));
    *(_QWORD *)a4 = v10;
    v8 = a1[26];
    if ( !v10 )
      goto LABEL_24;
    goto LABEL_10;
  }
LABEL_21:
  if ( (unsigned __int16)sub_3220AA0(v8) <= 4u )
    return sub_3737B50(a1, a2, a3, a4);
  v8 = a1[26];
  v10 = *(_QWORD **)a4;
  v11 = *(_DWORD *)(v8 + 3760);
  if ( !*(_QWORD *)a4 )
    goto LABEL_23;
LABEL_10:
  if ( (unsigned int)(v11 - 3) <= 1 && off_4C5D170 != (_UNKNOWN *)v10 )
  {
    v12 = sub_3222A80(v8, v10[1]);
    v8 = a1[26];
    v13 = v12;
    if ( v12 )
    {
      if ( a4 != v12 )
      {
        if ( *(_DWORD *)(v8 + 3760) == 3 )
        {
          v23 = sub_A777F0(0x10u, a1 + 11);
          v26 = (__int64 **)v23;
          if ( v23 )
          {
            *(_QWORD *)v23 = 0;
            *(_DWORD *)(v23 + 8) = 0;
          }
          sub_324B9D0(a1, (unsigned __int64 **)v23, a4, v24, v25);
          return sub_3249790(a1, a2, a3, 24, v26);
        }
        else
        {
          v29 = (unsigned int)sub_37291A0(v8 + 4840, v12, 0);
          v14 = (_QWORD *)sub_A777F0(0x18u, a1 + 11);
          v15 = a1 + 11;
          v16 = (__int64)v14;
          if ( v14 )
          {
            v14[1] = a4;
            v14[2] = v13;
            *v14 = v29;
          }
          v17 = (unsigned __int64 **)(a2 + 8);
          if ( !a3
            || (*(_BYTE *)(*(_QWORD *)(a1[23] + 200) + 904LL) & 0x40) == 0
            || (v30 = a1 + 11,
                v31 = v14,
                v27 = (unsigned __int16)sub_3220AA0(a1[26]),
                result = sub_E06A90(a3),
                v16 = (__int64)v31,
                v15 = v30,
                v27 >= (unsigned int)result) )
          {
            v33.m128i_i64[1] = v16;
            v33.m128i_i32[0] = 12;
            v33.m128i_i16[2] = a3;
            v33.m128i_i16[3] = 8193;
            return sub_3248F80(v17, v15, v33.m128i_i64);
          }
        }
        return result;
      }
    }
  }
LABEL_24:
  v19 = (unsigned int)sub_37291A0(v8 + 4840, a4, 0);
  v20 = sub_3220AA0(a1[26]);
  v21 = (unsigned __int64 **)(a2 + 8);
  v22 = v20 < 5u ? 7937 : 27;
  if ( !a3
    || (*(_BYTE *)(*(_QWORD *)(a1[23] + 200) + 904LL) & 0x40) == 0
    || (v32 = (unsigned __int64 **)(a2 + 8),
        v28 = (unsigned __int16)sub_3220AA0(a1[26]),
        result = sub_E06A90(a3),
        v21 = v32,
        v28 >= (unsigned int)result) )
  {
    v33.m128i_i16[2] = a3;
    v33.m128i_i16[3] = v22;
    v33.m128i_i64[1] = v19;
    v33.m128i_i32[0] = 1;
    return sub_3248F80(v21, a1 + 11, v33.m128i_i64);
  }
  return result;
}
