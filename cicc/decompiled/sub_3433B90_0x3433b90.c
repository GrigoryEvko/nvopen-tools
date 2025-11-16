// Function: sub_3433B90
// Address: 0x3433b90
//
void __fastcall sub_3433B90(__int64 a1, __int64 *a2, __int64 a3, __m128i a4)
{
  int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rsi
  unsigned __int8 *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r9
  __int64 v12; // rdx
  unsigned __int8 *v13; // r8
  unsigned __int8 **v14; // rdx
  unsigned __int8 *v15; // rax
  __int64 v16; // r9
  unsigned __int8 *v17; // rdx
  unsigned __int8 *v18; // r13
  __int64 v19; // rdx
  unsigned __int8 *v20; // r12
  unsigned __int8 **v21; // rdx
  __int64 v22; // rsi
  unsigned __int8 *v23; // [rsp+0h] [rbp-40h]
  __int64 v24; // [rsp+8h] [rbp-38h]
  __int64 v25; // [rsp+10h] [rbp-30h] BYREF
  int v26; // [rsp+18h] [rbp-28h]

  v6 = *((_DWORD *)a2 + 212);
  v7 = *a2;
  v25 = 0;
  v26 = v6;
  if ( v7 )
  {
    if ( &v25 != (__int64 *)(v7 + 48) )
    {
      v8 = *(_QWORD *)(v7 + 48);
      v25 = v8;
      if ( v8 )
        sub_B96E90((__int64)&v25, v8, 1);
    }
  }
  v9 = sub_3400BD0(a2[108], 2, (__int64)&v25, 8, 0, 1u, a4, 0);
  v11 = v10;
  v12 = *(unsigned int *)(a1 + 8);
  v13 = v9;
  if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    v23 = v9;
    v24 = v11;
    sub_C8D5F0(a1, (const void *)(a1 + 16), v12 + 1, 0x10u, (__int64)v9, v11);
    v12 = *(unsigned int *)(a1 + 8);
    v13 = v23;
    v11 = v24;
  }
  v14 = (unsigned __int8 **)(*(_QWORD *)a1 + 16 * v12);
  *v14 = v13;
  v14[1] = (unsigned __int8 *)v11;
  ++*(_DWORD *)(a1 + 8);
  v15 = sub_3400BD0(a2[108], a3, (__int64)&v25, 8, 0, 1u, a4, 0);
  v18 = v17;
  v19 = *(unsigned int *)(a1 + 8);
  v20 = v15;
  if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v19 + 1, 0x10u, v19 + 1, v16);
    v19 = *(unsigned int *)(a1 + 8);
  }
  v21 = (unsigned __int8 **)(*(_QWORD *)a1 + 16 * v19);
  *v21 = v20;
  v22 = v25;
  v21[1] = v18;
  ++*(_DWORD *)(a1 + 8);
  if ( v22 )
    sub_B91220((__int64)&v25, v22);
}
