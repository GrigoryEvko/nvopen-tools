// Function: sub_33FB4D0
// Address: 0x33fb4d0
//
unsigned __int8 *__fastcall sub_33FB4D0(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        __m128i a6)
{
  __int64 v9; // rax
  __int64 v10; // r9
  __int16 v11; // r15
  __int64 *v12; // rdi
  __int64 v13; // rax
  unsigned int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // r9
  unsigned int v17; // ebx
  unsigned __int8 *result; // rax
  __int64 v19; // rsi
  bool v20; // al
  __int64 v22; // [rsp+18h] [rbp-58h]
  __int64 v23; // [rsp+18h] [rbp-58h]
  unsigned __int8 *v24; // [rsp+18h] [rbp-58h]
  __int64 v25; // [rsp+18h] [rbp-58h]
  __int16 v26; // [rsp+20h] [rbp-50h] BYREF
  __int64 v27; // [rsp+28h] [rbp-48h]
  __int64 v28; // [rsp+30h] [rbp-40h] BYREF
  int v29; // [rsp+38h] [rbp-38h]

  v9 = *(_QWORD *)(a4 + 48) + 16LL * a5;
  v10 = *(_QWORD *)(a1 + 16);
  v11 = *(_WORD *)v9;
  v12 = *(__int64 **)(a1 + 40);
  v22 = v10;
  v27 = *(_QWORD *)(v9 + 8);
  v26 = v11;
  v13 = sub_2E79000(v12);
  v14 = sub_2FE6750(v22, a2, a3, v13);
  v16 = v15;
  v17 = v14;
  if ( v11 == (_WORD)v14 )
  {
    if ( v27 == v15 || v11 )
      return (unsigned __int8 *)a4;
LABEL_12:
    v25 = v15;
    v20 = sub_30070B0((__int64)&v26);
    v16 = v25;
    if ( !v20 )
      goto LABEL_8;
    return (unsigned __int8 *)a4;
  }
  if ( !v11 )
    goto LABEL_12;
  if ( (unsigned __int16)(v11 - 17) <= 0xD3u )
    return (unsigned __int8 *)a4;
LABEL_8:
  v19 = *(_QWORD *)(a4 + 80);
  v28 = v19;
  if ( v19 )
  {
    v23 = v16;
    sub_B96E90((__int64)&v28, v19, 1);
    v16 = v23;
  }
  v29 = *(_DWORD *)(a4 + 72);
  result = sub_33FB310(a1, a4, a5, (__int64)&v28, v17, v16, a6);
  if ( v28 )
  {
    v24 = result;
    sub_B91220((__int64)&v28, v28);
    return v24;
  }
  return result;
}
