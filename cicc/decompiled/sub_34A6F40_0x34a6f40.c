// Function: sub_34A6F40
// Address: 0x34a6f40
//
__int64 __fastcall sub_34A6F40(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r9
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned int v12; // esi
  int v13; // eax
  __int64 *v14; // r15
  int v15; // eax
  __int64 v16; // r8
  __int64 v17; // rdx
  __int64 v18; // rcx
  __m128i v19; // xmm0
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 *v23; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v24; // [rsp+8h] [rbp-38h] BYREF

  if ( (unsigned __int8)sub_34A25D0(a2, (__int64 *)a3, &v23) )
  {
    v8 = *(unsigned int *)(a2 + 24);
    *(_QWORD *)a1 = a2;
    *(_BYTE *)(a1 + 32) = 0;
    v9 = *(_QWORD *)(a2 + 8) + 56 * v8;
    v10 = *(_QWORD *)a2;
    *(_QWORD *)(a1 + 24) = v9;
    *(_QWORD *)(a1 + 8) = v10;
    *(_QWORD *)(a1 + 16) = v23;
    return a1;
  }
  v12 = *(_DWORD *)(a2 + 24);
  v13 = *(_DWORD *)(a2 + 16);
  v14 = v23;
  ++*(_QWORD *)a2;
  v15 = v13 + 1;
  v16 = 2 * v12;
  v24 = v14;
  if ( 4 * v15 >= 3 * v12 )
  {
    v12 *= 2;
  }
  else
  {
    v17 = v12 - *(_DWORD *)(a2 + 20) - v15;
    v18 = v12 >> 3;
    if ( (unsigned int)v17 > (unsigned int)v18 )
      goto LABEL_6;
  }
  sub_34A6CF0(a2, v12);
  sub_34A25D0(a2, (__int64 *)a3, &v24);
  v14 = v24;
  v15 = *(_DWORD *)(a2 + 16) + 1;
LABEL_6:
  *(_DWORD *)(a2 + 16) = v15;
  if ( *v14 != -4096 || v14[1] != -1 || v14[2] != -1 )
    --*(_DWORD *)(a2 + 20);
  *v14 = *(_QWORD *)a3;
  v19 = _mm_loadu_si128((const __m128i *)(a3 + 8));
  v14[3] = (__int64)(v14 + 5);
  v14[4] = 0x100000000LL;
  *(__m128i *)(v14 + 1) = v19;
  if ( *(_DWORD *)(a4 + 8) )
    sub_349D880((__int64)(v14 + 3), (char **)a4, v17, v18, v16, v7);
  v20 = *(unsigned int *)(a2 + 24);
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v14;
  *(_BYTE *)(a1 + 32) = 1;
  v21 = *(_QWORD *)(a2 + 8) + 56 * v20;
  v22 = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 24) = v21;
  *(_QWORD *)(a1 + 8) = v22;
  return a1;
}
