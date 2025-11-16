// Function: sub_1F7FE60
// Address: 0x1f7fe60
//
__int64 __fastcall sub_1F7FE60(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 result; // rax
  unsigned __int8 *v6; // rax
  __int64 *v7; // rdi
  unsigned int v8; // ecx
  __int64 v9; // r8
  _QWORD *v10; // rbx
  __int64 v11; // [rsp+0h] [rbp-30h] BYREF
  int v12; // [rsp+8h] [rbp-28h]

  if ( *(_WORD *)(**(_QWORD **)(a2 + 32) + 24LL) == 48 )
  {
    v6 = *(unsigned __int8 **)(a2 + 40);
    v7 = *(__int64 **)a1;
    v8 = *v6;
    v9 = *((_QWORD *)v6 + 1);
    v11 = 0;
    v12 = 0;
    v10 = sub_1D2B300(v7, 0x30u, (__int64)&v11, v8, v9, a2);
    if ( v11 )
      sub_161E7C0((__int64)&v11, v11);
    return (__int64)v10;
  }
  else
  {
    result = sub_1F7F730(a2, *(_QWORD *)(a1 + 8), *(__int64 **)a1, *(_BYTE *)(a1 + 25), *(_BYTE *)(a1 + 24), a3, a4, a5);
    if ( !result )
      return 0;
  }
  return result;
}
