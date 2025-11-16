// Function: sub_12319F0
// Address: 0x12319f0
//
__int64 __fastcall sub_12319F0(__int64 a1, _QWORD *a2, __int64 *a3)
{
  unsigned __int8 v3; // r13
  int v5; // eax
  __int16 v6; // r14
  unsigned __int64 v7; // r15
  const char *v9; // rax
  unsigned __int64 v10; // rsi
  __int64 v11; // rax
  const char *v12; // rax
  int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rdx
  int v19; // edx
  bool v20; // zf
  int v21; // eax
  int v22; // r15d
  _QWORD *v23; // rax
  _QWORD *v24; // rbx
  bool v25; // cf
  unsigned __int64 v26; // [rsp+0h] [rbp-A0h]
  char v27; // [rsp+1Ch] [rbp-84h] BYREF
  char v28; // [rsp+1Dh] [rbp-83h] BYREF
  __int16 v29; // [rsp+1Eh] [rbp-82h] BYREF
  unsigned int v30; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v31; // [rsp+24h] [rbp-7Ch] BYREF
  __int64 v32; // [rsp+28h] [rbp-78h] BYREF
  __int64 v33; // [rsp+30h] [rbp-70h] BYREF
  __int64 v34; // [rsp+38h] [rbp-68h] BYREF
  _QWORD v35[4]; // [rsp+40h] [rbp-60h] BYREF
  char v36; // [rsp+60h] [rbp-40h]
  char v37; // [rsp+61h] [rbp-3Fh]

  v3 = 0;
  v29 = 0;
  v5 = *(_DWORD *)(a1 + 240);
  v27 = 0;
  v30 = 0;
  v31 = 0;
  v28 = 1;
  if ( v5 == 32 )
  {
    v3 = 1;
    v5 = sub_1205200(a1 + 176);
    *(_DWORD *)(a1 + 240) = v5;
  }
  v6 = 0;
  if ( v5 == 68 )
  {
    v6 = 1;
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  }
  v7 = *(_QWORD *)(a1 + 232);
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v32, a3) )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' after cmpxchg address") )
    return 1;
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v33, a3) )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' after cmpxchg cmp operand") )
    return 1;
  v26 = *(_QWORD *)(a1 + 232);
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v34, a3)
    || (unsigned __int8)sub_120E4B0(a1, 1u, &v28, &v30)
    || (unsigned __int8)sub_120E3E0(a1, &v31)
    || (unsigned __int8)sub_120DF00(a1, &v29, &v27) )
  {
    return 1;
  }
  if ( v30 <= 1 )
  {
    v37 = 1;
    v9 = "invalid cmpxchg success ordering";
    goto LABEL_20;
  }
  if ( v31 - 5 <= 1 || v31 <= 1 )
  {
    v37 = 1;
    v9 = "invalid cmpxchg failure ordering";
LABEL_20:
    v10 = *(_QWORD *)(a1 + 232);
    v35[0] = v9;
    v36 = 3;
    sub_11FD800(a1 + 176, v10, (__int64)v35, 1);
    return 1;
  }
  if ( *(_BYTE *)(*(_QWORD *)(v32 + 8) + 8LL) != 14 )
  {
    v37 = 1;
    v36 = 3;
    v35[0] = "cmpxchg operand must be a pointer";
    sub_11FD800(a1 + 176, v7, (__int64)v35, 1);
    return 1;
  }
  v11 = *(_QWORD *)(v34 + 8);
  if ( v11 != *(_QWORD *)(v33 + 8) )
  {
    v37 = 1;
    v12 = "compare value and new value type do not match";
LABEL_24:
    v35[0] = v12;
    v36 = 3;
    sub_11FD800(a1 + 176, v26, (__int64)v35, 1);
    return 1;
  }
  v13 = *(unsigned __int8 *)(v11 + 8);
  if ( v13 == 13 || v13 == 7 )
  {
    v37 = 1;
    v12 = "cmpxchg operand must be a first class value";
    goto LABEL_24;
  }
  v14 = sub_B2BEC0(a3[1]);
  v15 = sub_9C6480(v14, *(_QWORD *)(v33 + 8));
  v35[1] = v16;
  v35[0] = v15;
  v17 = sub_CA1930(v35);
  _BitScanReverse64(&v18, v17);
  v19 = v18 ^ 0x3F;
  v20 = v17 == 0;
  v21 = 64;
  if ( !v20 )
    v21 = v19;
  if ( HIBYTE(v29) )
    v22 = (unsigned __int8)v29;
  else
    v22 = 63 - v21;
  v23 = sub_BD2C40(80, unk_3F148C4);
  v24 = v23;
  if ( v23 )
    sub_B4D5A0((__int64)v23, v32, v33, v34, v22, v30, v31, v28, 0, 0);
  v25 = v27 == 0;
  *((_WORD *)v24 + 1) = *((_WORD *)v24 + 1) & 0xFFFC | v6 | (2 * v3);
  *a2 = v24;
  return v25 ? 0 : 2;
}
