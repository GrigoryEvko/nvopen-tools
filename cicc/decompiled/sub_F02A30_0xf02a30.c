// Function: sub_F02A30
// Address: 0xf02a30
//
__int64 __fastcall sub_F02A30(__int64 **a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rbx
  __int64 v5; // r13
  void *v6; // rdi
  unsigned __int64 v7; // r14
  void *v8; // rsi
  __int64 v9; // [rsp+0h] [rbp-40h] BYREF
  __int64 v10; // [rsp+8h] [rbp-38h]
  __int64 v11; // [rsp+10h] [rbp-30h]

  result = (__int64)sub_F02910(&v9, a1);
  v4 = v9;
  v5 = v10;
  if ( v9 != v10 )
  {
    do
    {
      v6 = *(void **)(a2 + 32);
      v7 = *(_QWORD *)(v4 + 8);
      v8 = *(void **)v4;
      if ( v7 <= *(_QWORD *)(a2 + 24) - (_QWORD)v6 )
      {
        if ( v7 )
        {
          memcpy(v6, v8, *(_QWORD *)(v4 + 8));
          *(_QWORD *)(a2 + 32) += v7;
        }
      }
      else
      {
        sub_CB6200(a2, (unsigned __int8 *)v8, *(_QWORD *)(v4 + 8));
      }
      v4 += 16;
      result = sub_CB5D20(a2, 0);
    }
    while ( v5 != v4 );
    v5 = v9;
  }
  if ( v5 )
    return j_j___libc_free_0(v5, v11 - v5);
  return result;
}
