// Function: sub_3422DF0
// Address: 0x3422df0
//
unsigned __int64 __fastcall sub_3422DF0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7)
{
  __int64 v10; // rax
  unsigned __int64 result; // rax
  __int64 v13; // rdx
  unsigned __int8 *v14; // r8
  __int64 v15; // rax
  __int64 v16; // r9
  unsigned __int8 **v17; // rax
  __int64 v18; // rax
  unsigned __int16 *v19; // r12
  __int64 v20; // rsi
  unsigned __int8 *v21; // rax
  __int64 v22; // r9
  unsigned __int8 *v23; // rdx
  unsigned __int8 *v24; // r13
  __int64 v25; // rdx
  unsigned __int8 *v26; // r12
  unsigned __int8 **v27; // rdx
  unsigned __int8 *v28; // [rsp+0h] [rbp-40h]
  __int64 v29; // [rsp+8h] [rbp-38h]

  if ( *(_DWORD *)(a3 + 24) == 11 )
  {
    v14 = sub_3400BD0(*(_QWORD *)(a1 + 64), 2, a5, 8, 0, 1u, a7, 0);
    v15 = *(unsigned int *)(a2 + 8);
    v16 = v13;
    if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      v28 = v14;
      v29 = v13;
      sub_C8D5F0(a2, (const void *)(a2 + 16), v15 + 1, 0x10u, (__int64)v14, v13);
      v15 = *(unsigned int *)(a2 + 8);
      v14 = v28;
      v16 = v29;
    }
    v17 = (unsigned __int8 **)(*(_QWORD *)a2 + 16 * v15);
    *v17 = v14;
    v17[1] = (unsigned __int8 *)v16;
    ++*(_DWORD *)(a2 + 8);
    v18 = *(_QWORD *)(a3 + 96);
    v19 = (unsigned __int16 *)(*(_QWORD *)(a3 + 48) + 16LL * (unsigned int)a4);
    if ( *(_DWORD *)(v18 + 32) <= 0x40u )
      v20 = *(_QWORD *)(v18 + 24);
    else
      v20 = **(_QWORD **)(v18 + 24);
    v21 = sub_3400BD0(*(_QWORD *)(a1 + 64), v20, a5, *v19, *((_QWORD *)v19 + 1), 1u, a7, 0);
    v24 = v23;
    v25 = *(unsigned int *)(a2 + 8);
    v26 = v21;
    result = *(unsigned int *)(a2 + 12);
    if ( v25 + 1 > result )
    {
      result = sub_C8D5F0(a2, (const void *)(a2 + 16), v25 + 1, 0x10u, v25 + 1, v22);
      v25 = *(unsigned int *)(a2 + 8);
    }
    v27 = (unsigned __int8 **)(*(_QWORD *)a2 + 16 * v25);
    *v27 = v26;
    v27[1] = v24;
    ++*(_DWORD *)(a2 + 8);
  }
  else
  {
    v10 = *(unsigned int *)(a2 + 8);
    if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), v10 + 1, 0x10u, a5, a6);
      v10 = *(unsigned int *)(a2 + 8);
    }
    result = *(_QWORD *)a2 + 16 * v10;
    *(_QWORD *)result = a3;
    *(_QWORD *)(result + 8) = a4;
    ++*(_DWORD *)(a2 + 8);
  }
  return result;
}
