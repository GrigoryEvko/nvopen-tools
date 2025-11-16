// Function: sub_CA9F50
// Address: 0xca9f50
//
__int64 __fastcall sub_CA9F50(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v9; // rbx

  result = sub_22077B0(344);
  v9 = result;
  if ( result )
    result = sub_CA9E40(result, a2, a3, a4, a5, a6);
  *a1 = v9;
  a1[1] = 0;
  return result;
}
