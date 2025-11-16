// Function: sub_3815150
// Address: 0x3815150
//
unsigned __int8 *__fastcall sub_3815150(__int64 *a1, __int64 a2)
{
  __int64 (__fastcall *v3)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v4; // rax
  unsigned __int16 v5; // si
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // r9
  unsigned int v9; // r13d
  __int64 v10; // r15
  __int64 v11; // rsi
  _QWORD *v12; // rdi
  unsigned int v13; // esi
  unsigned __int8 *v14; // r12
  __int64 v16; // rdx
  __int64 v17; // [rsp+0h] [rbp-50h] BYREF
  int v18; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+10h] [rbp-40h]

  v3 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  v7 = a1[1];
  if ( v3 == sub_2D56A50 )
  {
    HIWORD(v9) = 0;
    sub_2FE6CC0((__int64)&v17, *a1, *(_QWORD *)(v7 + 64), v5, v6);
    LOWORD(v9) = v18;
    v10 = v19;
  }
  else
  {
    v9 = v3(*a1, *(_QWORD *)(v7 + 64), v5, v6);
    v10 = v16;
  }
  v11 = *(_QWORD *)(a2 + 80);
  v17 = v11;
  if ( v11 )
    sub_B96E90((__int64)&v17, v11, 1);
  v12 = (_QWORD *)a1[1];
  v13 = *(_DWORD *)(a2 + 24);
  v18 = *(_DWORD *)(a2 + 72);
  v14 = sub_3406EB0(
          v12,
          v13,
          (__int64)&v17,
          v9,
          v10,
          v8,
          *(_OWORD *)*(_QWORD *)(a2 + 40),
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
  if ( v17 )
    sub_B91220((__int64)&v17, v17);
  return v14;
}
