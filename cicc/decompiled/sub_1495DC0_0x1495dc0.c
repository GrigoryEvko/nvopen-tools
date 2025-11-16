// Function: sub_1495DC0
// Address: 0x1495dc0
//
__int64 __fastcall sub_1495DC0(__int64 a1, __m128i a2, __m128i a3)
{
  __int64 result; // rax
  _QWORD *v5; // r12
  _QWORD *v6; // r13
  unsigned __int64 v7; // rdi
  _QWORD v8[5]; // [rsp-108h] [rbp-108h] BYREF
  __int64 *v9; // [rsp-E0h] [rbp-E0h]
  __int64 v10; // [rsp-D0h] [rbp-D0h] BYREF
  _QWORD *v11; // [rsp-48h] [rbp-48h]
  unsigned int v12; // [rsp-38h] [rbp-38h]

  result = *(_QWORD *)(a1 + 352);
  if ( !result )
  {
    sub_14585E0((__int64)v8);
    *(_QWORD *)(a1 + 352) = sub_14959A0(*(_QWORD **)(a1 + 112), *(_QWORD *)(a1 + 120), (__int64)v8, a2, a3);
    sub_1495190(a1, (__int64)v8, a2, a3);
    v8[0] = &unk_49EC708;
    if ( v12 )
    {
      v5 = v11;
      v6 = &v11[7 * v12];
      do
      {
        if ( *v5 != -16 && *v5 != -8 )
        {
          v7 = v5[1];
          if ( (_QWORD *)v7 != v5 + 3 )
            _libc_free(v7);
        }
        v5 += 7;
      }
      while ( v6 != v5 );
    }
    j___libc_free_0(v11);
    if ( v9 != &v10 )
      _libc_free((unsigned __int64)v9);
    return *(_QWORD *)(a1 + 352);
  }
  return result;
}
