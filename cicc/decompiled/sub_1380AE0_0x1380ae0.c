// Function: sub_1380AE0
// Address: 0x1380ae0
//
_QWORD *__fastcall sub_1380AE0(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  __int64 v4; // rbx
  __int64 i; // r12
  __int64 v6; // rsi
  __int64 v7; // rdi
  _WORD *v8; // rdx
  _QWORD *result; // rax
  __int64 v10; // [rsp+0h] [rbp-60h] BYREF
  __int64 v11; // [rsp+8h] [rbp-58h]
  char v12; // [rsp+10h] [rbp-50h]
  __m128i *v13[2]; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v14[6]; // [rsp+30h] [rbp-30h] BYREF

  v10 = a1;
  v11 = a2;
  v12 = a3;
  sub_16E2FC0(v13, a4);
  sub_137F580(&v10, v13);
  v4 = *(_QWORD *)(*(_QWORD *)v11 + 80LL);
  for ( i = *(_QWORD *)v11 + 72LL; i != v4; v4 = *(_QWORD *)(v4 + 8) )
  {
    v6 = v4 - 24;
    if ( !v4 )
      v6 = 0;
    sub_137FA70(&v10, v6);
  }
  v7 = v10;
  v8 = *(_WORD **)(v10 + 24);
  if ( *(_QWORD *)(v10 + 16) - (_QWORD)v8 <= 1u )
  {
    sub_16E7EE0(v10, "}\n", 2);
  }
  else
  {
    *v8 = 2685;
    *(_QWORD *)(v7 + 24) += 2LL;
  }
  result = v14;
  if ( v13[0] != (__m128i *)v14 )
    return (_QWORD *)j_j___libc_free_0(v13[0], v14[0] + 1LL);
  return result;
}
