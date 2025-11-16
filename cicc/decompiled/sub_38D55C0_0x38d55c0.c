// Function: sub_38D55C0
// Address: 0x38d55c0
//
char *__fastcall sub_38D55C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  int v3; // r8d
  int v4; // r9d
  unsigned int v5; // eax
  __int64 v6; // rax
  __m128i *v7; // rax
  __int64 v8; // rdx
  char *v9; // rdi
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // r13
  int v12; // r12d
  char *result; // rax
  size_t v14; // r13
  __m128i v15; // [rsp+0h] [rbp-40h] BYREF
  __int64 v16; // [rsp+10h] [rbp-30h]

  v2 = sub_38D4BB0(a1, 0);
  sub_38D4150(a1, v2, *(unsigned int *)(v2 + 72));
  v5 = *(_DWORD *)(v2 + 72);
  v15.m128i_i64[0] = a2;
  v15.m128i_i64[1] = v5 | 0xA00000000LL;
  v6 = *(unsigned int *)(v2 + 120);
  v16 = 0;
  if ( (unsigned int)v6 >= *(_DWORD *)(v2 + 124) )
  {
    sub_16CD150(v2 + 112, (const void *)(v2 + 128), 0, 24, v3, v4);
    v6 = *(unsigned int *)(v2 + 120);
  }
  v7 = (__m128i *)(*(_QWORD *)(v2 + 112) + 24 * v6);
  v8 = v16;
  *v7 = _mm_loadu_si128(&v15);
  v7[1].m128i_i64[0] = v8;
  v9 = (char *)*(unsigned int *)(v2 + 72);
  v10 = *(unsigned int *)(v2 + 76);
  ++*(_DWORD *)(v2 + 120);
  v11 = (unsigned __int64)(v9 + 4);
  v12 = (int)v9;
  if ( v10 < (unsigned __int64)(v9 + 4) )
  {
    sub_16CD150(v2 + 64, (const void *)(v2 + 80), v11, 1, v3, v4);
    result = (char *)*(unsigned int *)(v2 + 72);
    v9 = result;
  }
  else
  {
    result = v9;
  }
  v14 = v11 - (_QWORD)result;
  if ( v14 )
    result = (char *)memset(&v9[*(_QWORD *)(v2 + 64)], 0, v14);
  *(_DWORD *)(v2 + 72) = v12 + 4;
  return result;
}
