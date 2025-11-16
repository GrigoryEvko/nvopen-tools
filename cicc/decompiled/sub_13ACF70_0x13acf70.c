// Function: sub_13ACF70
// Address: 0x13acf70
//
__int64 __fastcall sub_13ACF70(__int64 a1, char a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  char v8; // r8
  __int64 result; // rax
  __int64 v10; // rcx

  *(_BYTE *)(a4 + 144LL * a3 + 136) = a2;
  v7 = sub_13ACDD0(a1, a4);
  if ( !v7 || (v8 = sub_13A7760(a1, 38, v7, a5), result = 0, !v8) )
  {
    v10 = sub_13ACEA0(a1, a4);
    result = 1;
    if ( v10 )
      return (unsigned int)sub_13A7760(a1, 38, a5, v10) ^ 1;
  }
  return result;
}
