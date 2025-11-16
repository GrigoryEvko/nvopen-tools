// Function: sub_37016C0
// Address: 0x37016c0
//
unsigned __int64 *__fastcall sub_37016C0(
        unsigned __int64 *a1,
        _QWORD *a2,
        __m128i *a3,
        const __m128i *a4,
        unsigned int a5)
{
  __int64 v8; // rdi
  __int64 v9; // r8
  __int64 v10; // rsi
  unsigned __int64 v11; // rax
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rbx
  unsigned __int64 v18; // [rsp+18h] [rbp-58h] BYREF
  _OWORD v19[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v20; // [rsp+40h] [rbp-30h]

  v8 = a2[7];
  if ( v8 && !a2[5] && !a2[6] )
  {
    if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v8 + 40LL))(v8) )
    {
      v13 = a4[2].m128i_i64[0];
      v19[0] = _mm_loadu_si128(a4);
      v20 = v13;
      v19[1] = _mm_loadu_si128(a4 + 1);
      if ( (unsigned __int8)v13 > 1u )
        (*(void (__fastcall **)(_QWORD, _OWORD *))(*(_QWORD *)a2[7] + 24LL))(a2[7], v19);
    }
    (**(void (__fastcall ***)(_QWORD, __m128i *, __int64))a2[7])(a2[7], a3, 16);
    if ( a2[7] && !a2[5] && !a2[6] )
      a2[8] += 16LL;
    goto LABEL_8;
  }
  if ( (unsigned int)sub_3700ED0((__int64)a2, (__int64)a2, (__int64)a3, (__int64)a4, a5) > 0xF )
  {
    v9 = a2[6];
    v10 = a2[5];
    if ( !v9 || a2[7] || v10 )
    {
      v19[0] = 0u;
      sub_1254950(&v18, v10, (__int64)v19, 0x10u);
      v11 = v18 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
LABEL_11:
        *a1 = v11 | 1;
        return a1;
      }
      *a3 = _mm_loadu_si128(*(const __m128i **)&v19[0]);
    }
    else
    {
      sub_3719260(v19, a2[6], a3, 16);
      v11 = *(_QWORD *)&v19[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (*(_QWORD *)&v19[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_11;
    }
LABEL_8:
    *a1 = 1;
    return a1;
  }
  v14 = sub_37F93E0();
  LOWORD(v20) = 257;
  v15 = sub_22077B0(0x40u);
  v16 = v15;
  if ( v15 )
  {
    sub_C63E60(v15, 2, v14, (__int64)v19);
    *(_QWORD *)v16 = &unk_4A3C5B0;
  }
  *a1 = v16 | 1;
  return a1;
}
