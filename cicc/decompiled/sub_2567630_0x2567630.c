// Function: sub_2567630
// Address: 0x2567630
//
__int64 __fastcall sub_2567630(__int64 a1, const __m128i *a2, __int64 a3, int a4, char a5)
{
  __m128i v9; // xmm0
  _QWORD *v10; // rax
  __int64 v11; // r12
  __int64 (__fastcall *v12)(__int64); // rax
  _BYTE *v13; // rdi
  __int64 (__fastcall *v14)(__int64); // rax
  char v15; // al
  __int64 (__fastcall *v16)(__int64); // rax
  _BYTE *v17; // rdi
  __int64 (__fastcall *v18)(__int64); // rax
  void *v21; // [rsp+0h] [rbp-50h] BYREF
  __m128i v22; // [rsp+8h] [rbp-48h]

  v9 = _mm_loadu_si128(a2);
  v21 = &unk_438FC87;
  v22 = v9;
  v10 = sub_25134D0(a1 + 136, (__int64 *)&v21);
  v11 = (__int64)v10;
  if ( v10 )
  {
    v11 = v10[3];
    if ( !v11 )
      return 0;
    if ( a4 != 2
      && a3
      && ((v12 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL), v12 != sub_2505F20)
        ? (v13 = (_BYTE *)v12(v11))
        : (v13 = (_BYTE *)(v11 + 88)),
          (v14 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v13 + 16LL), v14 != sub_2505E30)
        ? (v15 = ((__int64 (*)(void))v14)())
        : (v15 = v13[9]),
          v15) )
    {
      sub_250ED80(a1, v11, a3, a4);
      if ( a5 )
        return v11;
    }
    else if ( a5 )
    {
      return v11;
    }
    v16 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL);
    v17 = (_BYTE *)(v16 == sub_2505F20 ? v11 + 88 : v16(v11));
    v18 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v17 + 16LL);
    if ( !(v18 == sub_2505E30 ? v17[9] : ((__int64 (*)(void))v18)()) )
      return 0;
  }
  return v11;
}
