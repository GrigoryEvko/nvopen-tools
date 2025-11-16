// Function: sub_801720
// Address: 0x801720
//
_QWORD *__fastcall sub_801720(__int64 a1, __int64 a2, int a3, __int64 a4, __m128i *a5)
{
  __int64 v8; // r15
  __m128i *v9; // rdi
  _QWORD *result; // rax
  __int64 v12[4]; // [rsp+10h] [rbp-D0h] BYREF
  _BYTE v13[48]; // [rsp+30h] [rbp-B0h] BYREF
  __m128i v14; // [rsp+60h] [rbp-80h] BYREF
  char v15; // [rsp+73h] [rbp-6Dh]

  v8 = *(_QWORD *)(a1 + 24);
  if ( !v8 && (*(_BYTE *)(a1 + 9) & 8) != 0 )
    v8 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 152LL);
  if ( *(_BYTE *)(a1 + 8) != 2 || (*(_BYTE *)(*(_QWORD *)(a1 + 16) + 146LL) & 8) == 0 || *(_BYTE *)(v8 + 48) != 7 )
  {
    qword_4D03EA0 = a2;
    sub_7F90D0(a2, (__int64)&v14);
    sub_7F55E0(a1, (__int64)&v14, (__int64)v13);
    if ( a3 )
      v15 = 1;
    sub_7F9B80((__int64)v12);
    v9 = *(__m128i **)(a1 + 32);
    v12[0] = a1;
    if ( v9 )
      sub_7EE560(v9, 0);
    if ( (unsigned int)(a5->m128i_i32[0] - 3) > 2 )
      sub_7E2BA0((__int64)a5);
    sub_7FEC50(v8, &v14, v12, a4, 1, 0, a5, 0, 0);
    if ( (unsigned int)(a5->m128i_i32[0] - 3) > 2 )
      sub_7FAFA0((__int64)a5);
    result = &qword_4D03EA0;
    qword_4D03EA0 = 0;
  }
  return result;
}
