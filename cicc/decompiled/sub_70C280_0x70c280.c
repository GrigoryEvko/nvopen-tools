// Function: sub_70C280
// Address: 0x70c280
//
int *__fastcall sub_70C280(
        unsigned __int8 a1,
        const __m128i *a2,
        const __m128i *a3,
        __m128i *a4,
        int *a5,
        unsigned int *a6)
{
  int v8; // r12d
  int v9; // r12d
  int v10; // r12d
  const __m128i *v11; // r10
  int v12; // eax
  int v13; // r12d
  int v14; // r12d
  __m128i *v15; // rcx
  const __m128i *v16; // rsi
  int v17; // r12d
  __m128i *v18; // r14
  int v19; // r12d
  int v20; // r12d
  int v21; // r12d
  int v22; // r12d
  int *result; // rax
  const __m128i *v24; // [rsp+0h] [rbp-A0h]
  const __m128i *v26; // [rsp+28h] [rbp-78h]
  int v29; // [rsp+48h] [rbp-58h] BYREF
  unsigned int v30; // [rsp+4Ch] [rbp-54h] BYREF
  __m128i v31; // [rsp+50h] [rbp-50h] BYREF
  __m128i v32[4]; // [rsp+60h] [rbp-40h] BYREF

  sub_70BBE0(a1, a3, a3, v32, &v29, &v30);
  v8 = v29;
  *a6 = v30;
  v26 = a3 + 1;
  sub_70BBE0(a1, v26, v26, &v31, &v29, &v30);
  *a6 |= v30;
  v9 = v29 | v8;
  sub_70B8D0(a1, v32, &v31, v32, &v29, &v30);
  *a6 |= v30;
  v10 = v29 | v9;
  v11 = a3;
  if ( !unk_4D04248 && (v12 = sub_70B8A0(a1, v32), v11 = a3, v12) )
  {
    *a5 = 1;
    return a5;
  }
  else
  {
    v24 = v11;
    sub_70BBE0(a1, a2, v11, a4, &v29, &v30);
    *a6 |= v30;
    v13 = v29 | v10;
    sub_70BBE0(a1, a2 + 1, v26, &v31, &v29, &v30);
    *a6 |= v30;
    v14 = v29 | v13;
    sub_70B8D0(a1, a4, &v31, a4, &v29, &v30);
    *a6 |= v30;
    v15 = a4;
    v16 = a4;
    v17 = v29 | v14;
    v18 = a4 + 1;
    sub_70BCF0(a1, v16, v32, v15, &v29, &v30);
    *a6 |= v30;
    v19 = v29 | v17;
    sub_70BBE0(a1, a2, v26, v18, &v29, &v30);
    *a6 |= v30;
    v20 = v29 | v19;
    sub_70BBE0(a1, a2 + 1, v24, &v31, &v29, &v30);
    *a6 |= v30;
    v21 = v29 | v20;
    sub_70B9E0(a1, &v31, v18, v18, &v29, &v30);
    *a6 |= v30;
    v22 = v29 | v21;
    sub_70BCF0(a1, v18, v32, v18, &v29, &v30);
    *a5 = v29 | v22;
    result = (int *)v30;
    *a6 |= v30;
  }
  return result;
}
