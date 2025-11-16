// Function: sub_8CC1D0
// Address: 0x8cc1d0
//
__int64 ***__fastcall sub_8CC1D0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  __int64 v3; // r13
  __int64 v4; // r14
  __int64 v5; // rdx
  __int64 v6; // rcx
  _UNKNOWN *__ptr32 *v7; // r8
  _QWORD *v8; // rax
  __int64 ***result; // rax
  __int64 v10; // rsi

  v1 = *(_QWORD *)(a1 + 96);
  v2 = sub_8C9880(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v1 + 32) + 88LL) + 104LL));
  v3 = *(_QWORD *)(a1 + 88);
  v4 = *(_QWORD *)(*(_QWORD *)v2 + 88LL);
  v8 = sub_8C6FA0(*(_QWORD **)(v4 + 112), *(_QWORD *)(*(_QWORD *)(v1 + 24) + 88LL), v5, v6, v7);
  if ( !v8 )
    return (__int64 ***)sub_8CA1D0(v4, *(_QWORD *)(v1 + 24));
  result = (__int64 ***)v8[1];
  v10 = (__int64)result[11];
  if ( v10 != v3 )
  {
    result = *(__int64 ****)(v10 + 32);
    if ( !result )
      return sub_8CC0D0(v3, v10);
    v10 = (__int64)*result;
    if ( (__int64 **)v3 != *result )
      return sub_8CC0D0(v3, v10);
  }
  return result;
}
