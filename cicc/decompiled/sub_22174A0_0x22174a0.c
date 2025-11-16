// Function: sub_22174A0
// Address: 0x22174a0
//
__int64 __fastcall sub_22174A0(__int64 a1, wint_t a2, unsigned int a3)
{
  unsigned int v4; // ebp
  __int64 result; // rax

  if ( a2 <= 0x7F && *(_BYTE *)(a1 + 24) )
    return *(unsigned __int8 *)(a1 + (int)a2 + 25);
  __uselocale();
  v4 = wctob(a2);
  __uselocale();
  result = a3;
  if ( v4 != -1 )
    return v4;
  return result;
}
