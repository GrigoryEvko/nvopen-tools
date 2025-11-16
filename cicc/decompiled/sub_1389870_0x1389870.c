// Function: sub_1389870
// Address: 0x1389870
//
__int64 __fastcall sub_1389870(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 **v4; // rax
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 **v7; // rax
  __int64 *v8; // r15
  __int64 result; // rax
  __int64 *v10; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-28h]

  v3 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v4 = *(__int64 ***)(a2 - 8);
  else
    v4 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v5 = **v4;
  if ( *(_BYTE *)(v5 + 8) == 16 )
    v5 = **(_QWORD **)(v5 + 16);
  v11 = 8 * sub_15A9520(v3, *(_DWORD *)(v5 + 8) >> 8);
  if ( v11 > 0x40 )
    sub_16A4EF0(&v10, 0, 0);
  else
    v10 = 0;
  v6 = 0x7FFFFFFFFFFFFFFFLL;
  if ( (unsigned __int8)sub_1634900(a2, *(_QWORD *)(a1 + 8), &v10) )
  {
    if ( v11 > 0x40 )
      v6 = *v10;
    else
      v6 = (__int64)((_QWORD)v10 << (64 - (unsigned __int8)v11)) >> (64 - (unsigned __int8)v11);
  }
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v7 = *(__int64 ***)(a2 - 8);
  else
    v7 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v8 = *v7;
  result = **v7;
  if ( *(_BYTE *)(result + 8) == 15 )
  {
    result = *(_QWORD *)a2;
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
    {
      result = sub_1389430(a1, (__int64)v8, 0);
      if ( v8 != (__int64 *)a2 )
        result = (__int64)sub_1389510(a1, (__int64)v8, a2, (__m128i *)v6);
    }
  }
  if ( v11 > 0x40 )
  {
    if ( v10 )
      return j_j___libc_free_0_0(v10);
  }
  return result;
}
