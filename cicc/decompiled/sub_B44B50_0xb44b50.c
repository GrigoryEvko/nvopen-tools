// Function: sub_B44B50
// Address: 0xb44b50
//
__int64 __fastcall sub_B44B50(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rsi
  __int64 i; // rbx
  __int64 v6; // r12
  __int64 v7; // rdi
  _QWORD v8[3]; // [rsp-58h] [rbp-58h] BYREF
  int v9; // [rsp-40h] [rbp-40h] BYREF
  __int64 v10; // [rsp-38h] [rbp-38h]
  int *v11; // [rsp-30h] [rbp-30h]
  int *v12; // [rsp-28h] [rbp-28h]
  __int64 v13; // [rsp-20h] [rbp-20h]

  result = (unsigned int)*(unsigned __int8 *)a1 - 34;
  if ( (unsigned __int8)(*(_BYTE *)a1 - 34) <= 0x33u )
  {
    v3 = 0x8000000000041LL;
    if ( _bittest64(&v3, result) )
    {
      v11 = &v9;
      v12 = &v9;
      v8[1] = 0x200400000LL;
      v9 = 0;
      v10 = 0;
      v13 = 0;
      v8[0] = 0x80000000000LL;
      v4 = sub_BD5C60(a1, a2);
      result = sub_A7A440(a1 + 9, (__int64 *)v4, 0, (__int64)v8);
      a1[9] = result;
      for ( i = v10; i; result = j_j___libc_free_0(v6, 88) )
      {
        v6 = i;
        sub_B43850(*(_QWORD **)(i + 24), v4);
        v7 = *(_QWORD *)(i + 32);
        i = *(_QWORD *)(i + 16);
        if ( v7 != v6 + 56 )
          _libc_free(v7, v4);
        v4 = 88;
      }
    }
  }
  return result;
}
