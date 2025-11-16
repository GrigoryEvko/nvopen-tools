// Function: sub_33DF4A0
// Address: 0x33df4a0
//
_BOOL8 __fastcall sub_33DF4A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  bool v8; // r8
  _BOOL8 result; // rax
  unsigned int v10; // r8d

  v8 = sub_33CF170(a4);
  result = 0;
  if ( !v8 )
  {
    v10 = sub_33D4D80(a1, a2, a3, 0);
    result = 1;
    if ( v10 > 1 )
      return (unsigned int)sub_33D4D80(a1, a4, a5, 0) <= 1;
  }
  return result;
}
