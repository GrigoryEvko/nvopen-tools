// Function: sub_E5C2E0
// Address: 0xe5c2e0
//
__int64 __fastcall sub_E5C2E0(__int64 a1, __int64 a2, unsigned __int8 a3, _QWORD *a4)
{
  __int64 v7; // rsi
  unsigned int v8; // r15d
  __int64 v10; // rdi
  unsigned int v11; // r13d
  __int64 v12; // r12
  __m128i *v13; // r12
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  const __m128i *v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v24; // [rsp+8h] [rbp-D8h]
  __int64 v25; // [rsp+10h] [rbp-D0h]
  int v26; // [rsp+18h] [rbp-C8h]
  _QWORD v27[4]; // [rsp+20h] [rbp-C0h] BYREF
  __int16 v28; // [rsp+40h] [rbp-A0h]
  __m128i v29[2]; // [rsp+50h] [rbp-90h] BYREF
  char v30; // [rsp+70h] [rbp-70h]
  char v31; // [rsp+71h] [rbp-6Fh]
  _QWORD v32[4]; // [rsp+80h] [rbp-60h] BYREF
  __int16 v33; // [rsp+A0h] [rbp-40h]

  if ( (*(_BYTE *)(a2 + 9) & 0x70) == 0x20 )
  {
    *(_BYTE *)(a2 + 8) |= 8u;
    v10 = *(_QWORD *)(a2 + 24);
    v11 = a3;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    v8 = sub_E81960(v10, &v23, a1);
    if ( (_BYTE)v8 )
    {
      v12 = v25;
      if ( v23 )
      {
        if ( !(unsigned __int8)sub_E5C2E0(a1, *(_QWORD *)(v23 + 16), v11, v32) )
          return 0;
        v12 += v32[0];
      }
      if ( v24 )
      {
        if ( !(unsigned __int8)sub_E5C2E0(a1, *(_QWORD *)(v24 + 16), v11, v32) )
          return 0;
        v12 -= v32[0];
      }
      *a4 = v12;
      return v8;
    }
    v31 = 1;
    v29[0].m128i_i64[0] = (__int64)"'";
    v13 = (__m128i *)v27;
    v30 = 3;
    v14 = sub_E5B9B0(a2);
    v33 = 1283;
    v18 = (const __m128i *)v32;
    v32[0] = "unable to evaluate offset for variable '";
    v32[2] = v14;
    v32[3] = v19;
LABEL_16:
    sub_9C6370(v13, v18, v29, v15, v16, v17);
    sub_C64D30((__int64)v13, 1u);
  }
  v7 = *(_QWORD *)a2;
  if ( !v7 )
  {
    if ( !a3 )
      return 0;
    v20 = a2;
    v31 = 1;
    v29[0].m128i_i64[0] = (__int64)"'";
    v13 = (__m128i *)v32;
    v30 = 3;
    v21 = sub_E5B9B0(v20);
    v27[0] = "unable to evaluate offset to undefined symbol '";
    v18 = (const __m128i *)v27;
    v27[3] = v22;
    v28 = 1283;
    v27[2] = v21;
    goto LABEL_16;
  }
  v8 = 1;
  *a4 = *(_QWORD *)(a2 + 24) + sub_E5C2C0(a1, v7);
  return v8;
}
