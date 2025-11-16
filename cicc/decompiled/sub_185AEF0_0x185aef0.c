// Function: sub_185AEF0
// Address: 0x185aef0
//
__int64 __fastcall sub_185AEF0(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  const char *v4; // r13
  size_t v5; // rdx
  size_t v6; // rbx
  size_t v7; // rdx
  const char *v8; // rsi
  size_t v9; // r12
  int v10; // eax
  __int64 result; // rax

  v2 = sub_1649F00(*a1);
  v3 = sub_1649F00(*a2);
  v4 = sub_1649960(v2);
  v6 = v5;
  v8 = sub_1649960(v3);
  v9 = v7;
  if ( v6 <= v7 )
  {
    if ( !v6 || (v10 = memcmp(v4, v8, v6)) == 0 )
    {
      result = 0;
      if ( v6 == v9 )
        return result;
      return v6 < v9 ? -1 : 1;
    }
    return (v10 >> 31) | 1u;
  }
  result = 1;
  if ( v7 )
  {
    v10 = memcmp(v4, v8, v7);
    if ( !v10 )
      return v6 < v9 ? -1 : 1;
    return (v10 >> 31) | 1u;
  }
  return result;
}
