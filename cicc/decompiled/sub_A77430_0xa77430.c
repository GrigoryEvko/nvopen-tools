// Function: sub_A77430
// Address: 0xa77430
//
__int64 *__fastcall sub_A77430(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v4; // r14
  __int64 v6; // rbx
  __int64 *v7; // r13
  const void *v8; // rdi
  size_t v9; // rdx
  size_t v10; // r8
  bool v11; // cc
  size_t v12; // rdx
  int v13; // eax
  size_t v15; // [rsp+0h] [rbp-60h]
  void *s2; // [rsp+8h] [rbp-58h]
  size_t v17; // [rsp+10h] [rbp-50h]
  __int64 *v18; // [rsp+18h] [rbp-48h]
  __int64 v19[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = a2 - (_QWORD)a1;
  v4 = v3 >> 3;
  v18 = a1;
  if ( v3 > 0 )
  {
    while ( 1 )
    {
      v6 = v4 >> 1;
      v7 = &v18[v4 >> 1];
      v19[0] = *v7;
      s2 = *(void **)a3;
      v17 = *(_QWORD *)(a3 + 8);
      if ( !sub_A71840((__int64)v19) )
        goto LABEL_3;
      v8 = (const void *)sub_A71FD0(v19);
      v10 = v9;
      v11 = v9 <= v17;
      v12 = v17;
      if ( v11 )
        v12 = v10;
      if ( v12 && (v15 = v10, v13 = memcmp(v8, s2, v12), v10 = v15, v13) )
      {
        if ( v13 >= 0 )
          goto LABEL_10;
LABEL_3:
        v18 = v7 + 1;
        v4 = v4 - v6 - 1;
        if ( v4 <= 0 )
          return v18;
      }
      else
      {
        if ( v10 < v17 )
          goto LABEL_3;
LABEL_10:
        v4 >>= 1;
        if ( v6 <= 0 )
          return v18;
      }
    }
  }
  return v18;
}
