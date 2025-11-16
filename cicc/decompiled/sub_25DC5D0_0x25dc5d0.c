// Function: sub_25DC5D0
// Address: 0x25dc5d0
//
__int64 __fastcall sub_25DC5D0(unsigned __int8 **a1, unsigned __int8 **a2)
{
  unsigned __int8 *v2; // r12
  unsigned __int8 *v3; // r13
  const char *v4; // rax
  size_t v5; // rdx
  size_t v6; // r12
  const char *v7; // r14
  const char *v8; // rax
  size_t v9; // rdx
  size_t v10; // rbx
  int v11; // eax
  __int64 result; // rax

  v2 = sub_BD3990(*a1, (__int64)a2);
  v3 = sub_BD3990(*a2, (__int64)a2);
  v4 = sub_BD5D20((__int64)v2);
  v6 = v5;
  v7 = v4;
  v8 = sub_BD5D20((__int64)v3);
  v10 = v9;
  if ( v6 <= v9 )
    v9 = v6;
  if ( v9 )
  {
    v11 = memcmp(v7, v8, v9);
    if ( v11 )
      return (v11 >> 31) | 1u;
  }
  result = 0;
  if ( v6 != v10 )
    return v6 < v10 ? -1 : 1;
  return result;
}
