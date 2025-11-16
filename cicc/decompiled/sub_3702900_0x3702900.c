// Function: sub_3702900
// Address: 0x3702900
//
unsigned __int64 *__fastcall sub_3702900(unsigned __int64 *a1, _QWORD *a2, char *a3, const __m128i *a4)
{
  __int64 v6; // rdi
  __int64 v7; // r14
  __int64 v9; // rax
  __int64 v10; // rsi
  char v11; // bl
  void (*v12)(void); // rax
  unsigned __int64 v14; // [rsp+18h] [rbp-58h] BYREF
  _OWORD v15[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v16; // [rsp+40h] [rbp-30h]

  v6 = a2[7];
  v7 = a2[5];
  if ( v6 )
  {
    if ( !v7 && !a2[6] )
    {
      if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v6 + 40LL))(v6) )
      {
        v9 = a4[2].m128i_i64[0];
        v15[0] = _mm_loadu_si128(a4);
        v16 = v9;
        v15[1] = _mm_loadu_si128(a4 + 1);
        if ( (unsigned __int8)v9 > 1u )
          (*(void (__fastcall **)(_QWORD, _OWORD *))(*(_QWORD *)a2[7] + 24LL))(a2[7], v15);
      }
      (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(*(_QWORD *)a2[7] + 8LL))(a2[7], (unsigned __int8)*a3, 1);
      if ( a2[7] && !a2[5] && !a2[6] )
        ++a2[8];
      goto LABEL_5;
    }
  }
  else
  {
    v10 = a2[6];
    if ( v10 && !v7 )
    {
      v11 = *a3;
      v12 = *(void (**)(void))(**(_QWORD **)(v10 + 24) + 16LL);
      if ( (char *)v12 != (char *)sub_3700C70 )
        v12();
      LOBYTE(v15[0]) = v11;
      sub_3719260(a1, v10, v15, 1);
      return a1;
    }
  }
  v15[0] = 0u;
  sub_1254950(&v14, v7, (__int64)v15, 1u);
  if ( (v14 & 0xFFFFFFFFFFFFFFFELL) == 0 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(v7 + 24) + 16LL))(*(_QWORD *)(v7 + 24));
    *a3 = **(_BYTE **)&v15[0];
LABEL_5:
    *a1 = 1;
    return a1;
  }
  *a1 = v14 & 0xFFFFFFFFFFFFFFFELL | 1;
  return a1;
}
