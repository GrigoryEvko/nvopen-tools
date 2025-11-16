// Function: sub_3701560
// Address: 0x3701560
//
unsigned __int64 *__fastcall sub_3701560(
        unsigned __int64 *a1,
        _QWORD *a2,
        _QWORD *a3,
        const __m128i *a4,
        unsigned int a5)
{
  __int64 v7; // rdi
  __int64 v8; // rsi
  unsigned __int64 v9; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // r14
  __int64 v14; // r13
  __int64 v15; // rax
  _OWORD v18[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v19; // [rsp+30h] [rbp-30h]

  v7 = a2[7];
  v8 = a2[5];
  if ( v7 )
  {
    if ( !v8 && !a2[6] )
    {
      v13 = *a3;
      v14 = a3[1] + 1LL;
      if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v7 + 40LL))(v7) )
      {
        v15 = a4[2].m128i_i64[0];
        v18[0] = _mm_loadu_si128(a4);
        v19 = v15;
        v18[1] = _mm_loadu_si128(a4 + 1);
        if ( (unsigned __int8)v15 > 1u )
          (*(void (__fastcall **)(_QWORD, _OWORD *))(*(_QWORD *)a2[7] + 24LL))(a2[7], v18);
      }
      (**(void (__fastcall ***)(_QWORD, __int64, __int64))a2[7])(a2[7], v13, v14);
      if ( a2[7] && !a2[5] && !a2[6] )
        a2[8] += v14;
      goto LABEL_10;
    }
  }
  else if ( a2[6] && !v8 )
  {
    v11 = (unsigned int)sub_3700ED0((__int64)a2, 0, (__int64)a3, (__int64)a4, a5) - 1;
    v12 = a3[1];
    if ( v11 < v12 )
      v12 = v11;
    sub_37192F0(v18, a2[6], *a3, v12);
    v9 = *(_QWORD *)&v18[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (*(_QWORD *)&v18[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
      goto LABEL_10;
LABEL_4:
    *a1 = v9 | 1;
    return a1;
  }
  sub_1254A40((unsigned __int64 *)v18, v8, a3);
  v9 = *(_QWORD *)&v18[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (*(_QWORD *)&v18[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_4;
LABEL_10:
  *a1 = 1;
  return a1;
}
