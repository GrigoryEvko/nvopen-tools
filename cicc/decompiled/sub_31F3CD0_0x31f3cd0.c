// Function: sub_31F3CD0
// Address: 0x31f3cd0
//
__int64 __fastcall sub_31F3CD0(__int64 a1, __int64 a2)
{
  unsigned __int8 *v3; // rbx
  unsigned __int8 *v4; // r13
  unsigned __int64 v5; // rsi
  __int64 result; // rax

  v3 = *(unsigned __int8 **)(a1 + 8);
  v4 = v3 + 8;
  do
  {
    v5 = *v3++;
    result = sub_C7F500(a2, v5, 0, 2u, 1);
  }
  while ( v3 != v4 );
  return result;
}
