// Function: sub_38ACC60
// Address: 0x38acc60
//
__int64 __fastcall sub_38ACC60(__int64 a1, _QWORD *a2, __int64 *a3, double a4, double a5, double a6)
{
  __int16 v6; // r13
  int v8; // eax
  __int16 v9; // r14
  unsigned __int64 v10; // r15
  const char *v12; // rax
  unsigned __int64 v13; // rsi
  __int64 v14; // rax
  const char *v15; // rax
  int v16; // eax
  _QWORD *v17; // r12
  unsigned __int64 v18; // [rsp+8h] [rbp-98h]
  unsigned __int64 v19; // [rsp+10h] [rbp-90h]
  char v20; // [rsp+2Fh] [rbp-71h] BYREF
  int v21; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v22; // [rsp+34h] [rbp-6Ch] BYREF
  __int64 v23; // [rsp+38h] [rbp-68h] BYREF
  __int64 **v24; // [rsp+40h] [rbp-60h] BYREF
  _QWORD *v25; // [rsp+48h] [rbp-58h] BYREF
  _QWORD v26[2]; // [rsp+50h] [rbp-50h] BYREF
  char v27; // [rsp+60h] [rbp-40h]
  char v28; // [rsp+61h] [rbp-3Fh]

  v6 = 0;
  v8 = *(_DWORD *)(a1 + 64);
  v20 = 1;
  v21 = 0;
  v22 = 0;
  if ( v8 == 30 )
  {
    v6 = 1;
    v8 = sub_3887100(a1 + 8);
    *(_DWORD *)(a1 + 64) = v8;
  }
  v9 = 0;
  if ( v8 == 66 )
  {
    v9 = 1;
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  }
  v10 = *(_QWORD *)(a1 + 56);
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v23, a3, a4, a5, a6) )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' after cmpxchg address") )
    return 1;
  v19 = *(_QWORD *)(a1 + 56);
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v24, a3, a4, a5, a6) )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' after cmpxchg cmp operand") )
    return 1;
  v18 = *(_QWORD *)(a1 + 56);
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v25, a3, a4, a5, a6)
    || (unsigned __int8)sub_388CFF0(a1, 1u, &v20, &v21)
    || (unsigned __int8)sub_388CF30(a1, &v22) )
  {
    return 1;
  }
  if ( v21 == 1 || v22 == 1 )
  {
    v28 = 1;
    v12 = "cmpxchg cannot be unordered";
LABEL_20:
    v13 = *(_QWORD *)(a1 + 56);
    v26[0] = v12;
    v27 = 3;
    return (unsigned __int8)sub_38814C0(a1 + 8, v13, (__int64)v26);
  }
  if ( byte_42880A0[8 * v22 + v21] )
  {
    v28 = 1;
    v12 = "cmpxchg failure argument shall be no stronger than the success argument";
    goto LABEL_20;
  }
  if ( v22 - 5 <= 1 )
  {
    v28 = 1;
    v12 = "cmpxchg failure ordering cannot include release semantics";
    goto LABEL_20;
  }
  if ( *(_BYTE *)(*(_QWORD *)v23 + 8LL) != 15 )
  {
    v28 = 1;
    v27 = 3;
    v26[0] = "cmpxchg operand must be a pointer";
    return (unsigned __int8)sub_38814C0(a1 + 8, v10, (__int64)v26);
  }
  v14 = *(_QWORD *)(*(_QWORD *)v23 + 24LL);
  if ( *v24 != (__int64 *)v14 )
  {
    v28 = 1;
    v27 = 3;
    v26[0] = "compare value and pointer type do not match";
    return (unsigned __int8)sub_38814C0(a1 + 8, v19, (__int64)v26);
  }
  if ( v14 != *v25 )
  {
    v28 = 1;
    v15 = "new value and pointer type do not match";
LABEL_27:
    v26[0] = v15;
    v27 = 3;
    return (unsigned __int8)sub_38814C0(a1 + 8, v18, (__int64)v26);
  }
  v16 = *(unsigned __int8 *)(v14 + 8);
  if ( v16 == 12 || !v16 )
  {
    v28 = 1;
    v15 = "cmpxchg operand must be a first class value";
    goto LABEL_27;
  }
  v17 = sub_1648A60(64, 3u);
  if ( v17 )
    sub_15F99E0((__int64)v17, v23, v24, (__int64)v25, v21, v22, v20, 0);
  *((_WORD *)v17 + 9) = *((_WORD *)v17 + 9) & 0xFEFE | v9 | (v6 << 8);
  *a2 = v17;
  return 0;
}
