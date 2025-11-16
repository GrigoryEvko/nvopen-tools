// Function: sub_7DDD40
// Address: 0x7ddd40
//
_QWORD *__fastcall sub_7DDD40(__m128i *a1, _QWORD *a2, __int64 a3, _QWORD *a4, int a5)
{
  const __m128i *v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  _BYTE *v15; // r12
  _QWORD *v16; // r15
  __int64 v17; // rax
  const __m128i *v18; // rbx
  _QWORD *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  _QWORD *result; // rax
  __int64 v24; // rax
  __int64 v25; // [rsp+0h] [rbp-50h]
  __int64 v26; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int64 v27[7]; // [rsp+18h] [rbp-38h] BYREF

  v26 = 0;
  v8 = (const __m128i *)sub_724D50(6);
  if ( a1 )
  {
    v9 = sub_7DB130(a1, v27, &v26);
    v10 = sub_7DDA20(v9);
    sub_72D510(v10, (__int64)v8, 1);
    v11 = sub_7DC070();
    sub_70FEE0((__int64)v8, v11, v12, v13, v14);
  }
  else
  {
    v24 = sub_7DC070();
    sub_72BB40(v24, v8);
    v27[0] = 16;
  }
  if ( a5 )
    v27[0] |= 0x20u;
  v15 = sub_724D50(1);
  sub_72BBE0((__int64)v15, v27[0], unk_4F06870);
  v16 = sub_724D50(10);
  v17 = qword_4F188C0;
  v16[22] = v8;
  v16[16] = v17;
  v8[7].m128i_i64[1] = (__int64)v15;
  v18 = (const __m128i *)sub_724D50(6);
  v19 = sub_72BA30(unk_4F06870);
  v20 = sub_72D2E0(v19);
  if ( v26 )
  {
    v25 = v20;
    sub_72D510(v26, (__int64)v18, 1);
    sub_70FEE0((__int64)v18, v25, v21, v22, v25);
  }
  else
  {
    sub_72BB40(v20, v18);
  }
  result = a2;
  *((_QWORD *)v15 + 15) = v18;
  v16[23] = v18;
  if ( *a2 )
  {
    result = *(_QWORD **)a3;
    *(_QWORD *)(*(_QWORD *)a3 + 120LL) = v16;
  }
  else
  {
    *a2 = v16;
  }
  *(_QWORD *)a3 = v16;
  ++*a4;
  return result;
}
