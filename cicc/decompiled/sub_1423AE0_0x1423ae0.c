// Function: sub_1423AE0
// Address: 0x1423ae0
//
__int64 __fastcall sub_1423AE0(__int64 *a1)
{
  __int64 result; // rax
  __int64 v3; // r12
  __int64 v4; // r13
  _QWORD *v5; // r12
  __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // [rsp-30h] [rbp-30h]

  result = a1[41];
  if ( !result )
  {
    v3 = *a1;
    v4 = a1[1];
    result = sub_22077B0(2144);
    if ( result )
    {
      v8 = result;
      sub_1423A60(result, (__int64)a1, v3, v4);
      result = v8;
    }
    v5 = (_QWORD *)a1[41];
    a1[41] = result;
    if ( v5 )
    {
      v6 = v5[265];
      *v5 = &unk_49EB390;
      j___libc_free_0(v6);
      v7 = v5[6];
      if ( (_QWORD *)v7 != v5 + 8 )
        _libc_free(v7);
      j_j___libc_free_0(v5, 2144);
      return a1[41];
    }
  }
  return result;
}
