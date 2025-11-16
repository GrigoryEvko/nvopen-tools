// Function: sub_2130290
// Address: 0x2130290
//
unsigned __int64 __fastcall sub_2130290(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128i a5,
        double a6,
        __m128i a7)
{
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rsi
  char v14; // di
  int v15; // r9d
  __int64 v16; // r13
  __int64 v17; // rax
  unsigned int v18; // eax
  const void **v19; // rdx
  const void **v20; // r13
  __int64 v21; // r9
  int v22; // edx
  __int64 *v23; // rbx
  __int128 v24; // rax
  __int64 *v25; // rax
  __int64 v26; // rsi
  unsigned int v27; // edx
  unsigned __int64 result; // rax
  int v29; // [rsp+4h] [rbp-8Ch]
  unsigned int v30; // [rsp+8h] [rbp-88h]
  __int64 v31; // [rsp+30h] [rbp-60h] BYREF
  int v32; // [rsp+38h] [rbp-58h]
  unsigned int v33; // [rsp+40h] [rbp-50h] BYREF
  const void **v34; // [rsp+48h] [rbp-48h]
  const void **v35; // [rsp+50h] [rbp-40h]

  v11 = *(_QWORD *)(a2 + 72);
  v31 = v11;
  if ( v11 )
    sub_1623A60((__int64)&v31, v11, 2);
  v12 = a1[1];
  v13 = *a1;
  v32 = *(_DWORD *)(a2 + 64);
  sub_1F40D10(
    (__int64)&v33,
    v13,
    *(_QWORD *)(v12 + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v14 = (char)v34;
  LOBYTE(v33) = (_BYTE)v34;
  v34 = v35;
  if ( (_BYTE)v33 )
    v15 = sub_2127930(v14);
  else
    v15 = sub_1F58D40((__int64)&v33);
  v16 = *a1;
  v29 = v15;
  v17 = sub_1E0A0C0(*(_QWORD *)(a1[1] + 32));
  v18 = sub_1F40B60(v16, v33, (__int64)v34, v17, 1);
  v20 = v19;
  v30 = v18;
  *(_QWORD *)a3 = sub_1D2B300((_QWORD *)a1[1], 0x9Bu, (__int64)&v31, v33, (__int64)v34, v21);
  *(_DWORD *)(a3 + 8) = v22;
  v23 = (__int64 *)a1[1];
  *(_QWORD *)&v24 = sub_1D38BB0((__int64)v23, (unsigned int)(v29 - 1), (__int64)&v31, v30, v20, 0, a5, a6, a7, 0);
  v25 = sub_1D332F0(
          v23,
          123,
          (__int64)&v31,
          v33,
          v34,
          0,
          *(double *)a5.m128i_i64,
          a6,
          a7,
          *(_QWORD *)a3,
          *(_QWORD *)(a3 + 8),
          v24);
  v26 = v31;
  *(_QWORD *)a4 = v25;
  result = v27;
  *(_DWORD *)(a4 + 8) = v27;
  if ( v26 )
    return sub_161E7C0((__int64)&v31, v26);
  return result;
}
