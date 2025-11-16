// Function: sub_179D030
// Address: 0x179d030
//
unsigned __int8 *__fastcall sub_179D030(__int64 a1, __int64 *a2, __int64 a3, __int64 *a4, char a5, char a6)
{
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned __int8 *v12; // r12
  unsigned __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rdx
  bool v16; // zf
  __int64 v17; // rsi
  __int64 v18; // rsi
  unsigned __int8 *v19; // rsi
  unsigned __int64 *v21; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v22; // [rsp+18h] [rbp-58h] BYREF
  char v23[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v24; // [rsp+30h] [rbp-40h]

  v24 = 257;
  v10 = sub_15FB440(23, a2, a3, (__int64)v23, 0);
  v11 = *(_QWORD *)(a1 + 8);
  v12 = (unsigned __int8 *)v10;
  if ( v11 )
  {
    v21 = *(unsigned __int64 **)(a1 + 16);
    sub_157E9D0(v11 + 40, v10);
    v13 = *v21;
    v14 = *((_QWORD *)v12 + 3) & 7LL;
    *((_QWORD *)v12 + 4) = v21;
    v13 &= 0xFFFFFFFFFFFFFFF8LL;
    *((_QWORD *)v12 + 3) = v13 | v14;
    *(_QWORD *)(v13 + 8) = v12 + 24;
    *v21 = *v21 & 7 | (unsigned __int64)(v12 + 24);
  }
  sub_164B780((__int64)v12, a4);
  v16 = *(_QWORD *)(a1 + 80) == 0;
  v22 = v12;
  if ( v16 )
    sub_4263D6(v12, a4, v15);
  (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v22);
  v17 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    v22 = *(unsigned __int8 **)a1;
    sub_1623A60((__int64)&v22, v17, 2);
    v18 = *((_QWORD *)v12 + 6);
    if ( v18 )
      sub_161E7C0((__int64)(v12 + 48), v18);
    v19 = v22;
    *((_QWORD *)v12 + 6) = v22;
    if ( v19 )
      sub_1623210((__int64)&v22, v19, (__int64)(v12 + 48));
  }
  if ( a5 )
    sub_15F2310((__int64)v12, 1);
  if ( a6 )
    sub_15F2330((__int64)v12, 1);
  return v12;
}
