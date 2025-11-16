// Function: sub_722AB0
// Address: 0x722ab0
//
unsigned __int64 __fastcall sub_722AB0(unsigned __int8 *a1, int a2)
{
  int v2; // esi
  unsigned __int64 result; // rax
  __int64 v5; // rsi
  __int64 v6; // rdx

  if ( unk_4F06BA4 )
  {
    if ( a2 )
    {
      v2 = a2 - 1;
      result = 0;
      do
        result = a1[v2] | (result << dword_4F06BA0);
      while ( v2-- != 0 );
      return result;
    }
    return 0;
  }
  if ( !a2 )
    return 0;
  v5 = (__int64)&a1[a2 - 1 + 1];
  result = 0;
  do
  {
    v6 = *a1++;
    result = v6 | (result << dword_4F06BA0);
  }
  while ( (unsigned __int8 *)v5 != a1 );
  return result;
}
