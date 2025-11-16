// Function: sub_3827040
// Address: 0x3827040
//
void __fastcall sub_3827040(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, __int64 a6, __int64 a7)
{
  __int64 v11; // rsi
  unsigned __int16 *v12; // rdx
  bool v13; // zf
  _QWORD *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // [rsp+0h] [rbp-40h] BYREF
  int v18; // [rsp+8h] [rbp-38h]

  v11 = *(_QWORD *)(a2 + 80);
  v17 = v11;
  if ( v11 )
    sub_B96E90((__int64)&v17, v11, 1);
  v12 = *(unsigned __int16 **)(a2 + 48);
  v13 = *(_DWORD *)(a2 + 24) == 193;
  v14 = (_QWORD *)a1[1];
  v18 = *(_DWORD *)(a2 + 72);
  v15 = sub_340F900(
          v14,
          (unsigned int)!v13 + 195,
          (__int64)&v17,
          *v12,
          *((_QWORD *)v12 + 1),
          a7,
          *(_OWORD *)*(_QWORD *)(a2 + 40),
          *(_OWORD *)*(_QWORD *)(a2 + 40),
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
  sub_375BC20(a1, v15, v16, a3, a4, a5);
  if ( v17 )
    sub_B91220((__int64)&v17, v17);
}
