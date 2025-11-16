// Function: sub_A56340
// Address: 0xa56340
//
const __m128i *__fastcall sub_A56340(__int64 a1, __int64 a2)
{
  const __m128i *v2; // r12
  __int64 v4; // r13
  char v5; // r14
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 (__fastcall *v8)(__int64, __int64); // rax
  void (__fastcall *v9)(_BYTE *, __int64, __int64); // rax
  void (__fastcall *v10)(_BYTE *, __int64, __int64); // rax
  _BYTE v11[16]; // [rsp+0h] [rbp-40h] BYREF
  void (__fastcall *v12)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-30h]
  __int64 v13; // [rsp+18h] [rbp-28h]

  if ( !*(_BYTE *)(a1 + 16) )
    return *(const __m128i **)(a1 + 40);
  v4 = *(_QWORD *)(a1 + 24);
  v5 = *(_BYTE *)(a1 + 17);
  *(_BYTE *)(a1 + 16) = 0;
  v6 = sub_22077B0(400);
  v2 = (const __m128i *)v6;
  if ( v6 )
  {
    a2 = v4;
    sub_A55A10(v6, v4, v5);
  }
  v7 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 8) = v2;
  if ( v7 )
  {
    v8 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v7 + 8LL);
    if ( v8 == sub_A554F0 )
    {
      sub_A552A0(v7, a2);
      j_j___libc_free_0(v7, 400);
    }
    else
    {
      ((void (__fastcall *)(__int64))v8)(v7);
    }
    v2 = *(const __m128i **)(a1 + 8);
  }
  v9 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a1 + 64);
  *(_QWORD *)(a1 + 40) = v2;
  if ( v9 )
  {
    v12 = 0;
    v9(v11, a1 + 48, 2);
    v13 = *(_QWORD *)(a1 + 72);
    v12 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a1 + 64);
    sub_A56220(v2, (__int64)v11);
    if ( v12 )
      v12(v11, v11, 3);
    v2 = *(const __m128i **)(a1 + 40);
  }
  v10 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a1 + 96);
  if ( v10 )
  {
    v12 = 0;
    v10(v11, a1 + 80, 2);
    v13 = *(_QWORD *)(a1 + 104);
    v12 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a1 + 96);
    sub_A562B0(v2, (__int64)v11);
    if ( v12 )
      v12(v11, v11, 3);
    return *(const __m128i **)(a1 + 40);
  }
  return v2;
}
