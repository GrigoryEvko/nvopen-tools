// Function: sub_1623B10
// Address: 0x1623b10
//
__int64 __fastcall sub_1623B10(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 *i; // rbx
  __int64 v4; // r12
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 result; // rax

  v2 = 8LL * *(unsigned int *)(a1 + 8);
  for ( i = (__int64 *)(a1 - v2); (__int64 *)a1 != i; ++i )
  {
    v4 = *i;
    if ( *i )
    {
      sub_161E7C0((__int64)i, *i);
      *i = v4;
      a2 = v4;
      sub_1623A60((__int64)i, v4, a1 & 0xFFFFFFFFFFFFFFFDLL | 2);
    }
  }
  *(_BYTE *)(a1 + 1) = 0;
  sub_161EA20(a1);
  result = *(unsigned int *)(a1 + 12);
  if ( !(_DWORD)result )
    return sub_161F120(a1, a2, v5, v6, v7);
  return result;
}
