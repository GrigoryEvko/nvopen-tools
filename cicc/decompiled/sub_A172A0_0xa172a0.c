// Function: sub_A172A0
// Address: 0xa172a0
//
__int64 __fastcall sub_A172A0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rbx

  result = 0;
  if ( *(char *)(a1 + 7) < 0 )
  {
    v2 = sub_BD2BC0(a1);
    v4 = v2 + v3;
    if ( *(char *)(a1 + 7) >= 0 )
      return v4 >> 4;
    else
      return (unsigned int)((v4 - sub_BD2BC0(a1)) >> 4);
  }
  return result;
}
