// Function: sub_38D4DC0
// Address: 0x38d4dc0
//
__int64 __fastcall sub_38D4DC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v10; // rbx
  int v11; // r8d
  int v12; // r9d
  __int64 result; // rax
  unsigned int v14; // r14d
  __int64 v15; // rax
  __m128i *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v21; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v22; // [rsp+18h] [rbp-58h] BYREF
  char v23; // [rsp+1Ch] [rbp-54h]
  __m128i v24; // [rsp+20h] [rbp-50h] BYREF
  __int64 v25; // [rsp+30h] [rbp-40h]

  sub_38CF290(a2, &v21);
  v10 = sub_38D4BB0(a1, a7);
  sub_38D4150(a1, v10, *(unsigned int *)(v10 + 72));
  (*(void (__fastcall **)(unsigned int *, _QWORD, __int64, __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 264) + 8LL) + 40LL))(
    &v22,
    *(_QWORD *)(*(_QWORD *)(a1 + 264) + 8LL),
    a3,
    a4);
  result = 1;
  if ( v23 )
  {
    v14 = v22;
    if ( !a5 )
    {
      v18 = *(_QWORD *)(a1 + 8);
      v19 = sub_38BFA60(v18, 1);
      a5 = sub_38CF310(v19, 0, v18, 0);
    }
    v24.m128i_i64[0] = a5;
    v24.m128i_i64[1] = __PAIR64__(v14, v21);
    v25 = a6;
    v15 = *(unsigned int *)(v10 + 120);
    if ( (unsigned int)v15 >= *(_DWORD *)(v10 + 124) )
    {
      sub_16CD150(v10 + 112, (const void *)(v10 + 128), 0, 24, v11, v12);
      v15 = *(unsigned int *)(v10 + 120);
    }
    v16 = (__m128i *)(*(_QWORD *)(v10 + 112) + 24 * v15);
    v17 = v25;
    *v16 = _mm_loadu_si128(&v24);
    v16[1].m128i_i64[0] = v17;
    ++*(_DWORD *)(v10 + 120);
    return 0;
  }
  return result;
}
