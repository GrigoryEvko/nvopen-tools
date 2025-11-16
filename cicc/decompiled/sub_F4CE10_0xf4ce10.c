// Function: sub_F4CE10
// Address: 0xf4ce10
//
__int64 __fastcall sub_F4CE10(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        _BYTE *a7,
        size_t a8)
{
  __int64 v10; // rbx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 i; // r12
  __int64 v15; // rdi
  __int64 result; // rax
  __int64 v17; // [rsp+0h] [rbp-50h] BYREF
  __int64 v18; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+10h] [rbp-40h]
  unsigned int v20; // [rsp+18h] [rbp-38h]

  if ( a2 )
  {
    v17 = 0;
    v18 = 0;
    v10 = a3 + 24;
    v19 = 0;
    v20 = 0;
    sub_F4C4C0(a1, a2, (__int64)&v17, a7, a8, (__int64)a5);
    for ( i = *(_QWORD *)(a4 + 32); i != v10; v10 = *(_QWORD *)(v10 + 8) )
    {
      v15 = v10 - 24;
      if ( !v10 )
        v15 = 0;
      sub_F460A0(v15, &v17, a5, v11, v12, v13);
    }
    return sub_C7D6A0(v18, 16LL * v20, 8);
  }
  return result;
}
