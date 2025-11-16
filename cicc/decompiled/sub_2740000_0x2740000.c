// Function: sub_2740000
// Address: 0x2740000
//
__int64 __fastcall sub_2740000(__int64 a1, __int64 a2, unsigned __int8 *a3, unsigned __int8 *a4, char a5)
{
  __int64 v9; // [rsp+0h] [rbp-80h] BYREF
  char *v10; // [rsp+8h] [rbp-78h]
  int v11; // [rsp+10h] [rbp-70h]
  char v12; // [rsp+18h] [rbp-68h] BYREF

  sub_273DA80(a1, a3, *(_QWORD *)a2, *(_BYTE *)(a2 + 8), *(_QWORD *)(a2 + 16));
  sub_273DA80((__int64)&v9, a4, *(_QWORD *)a2, a5, *(_QWORD *)(a2 + 16));
  *(_QWORD *)a1 += v9;
  sub_2739AD0(a1 + 8, (__m128i *)(*(_QWORD *)(a1 + 8) + 24LL * *(unsigned int *)(a1 + 16)), v10, &v10[24 * v11]);
  if ( v10 != &v12 )
    _libc_free((unsigned __int64)v10);
  return a1;
}
