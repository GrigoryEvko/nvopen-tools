// Function: sub_16FE750
// Address: 0x16fe750
//
__int64 __fastcall sub_16FE750(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v9; // rbx

  result = sub_22077B0(352);
  v9 = result;
  if ( result )
    result = sub_16FE640(result, a2, a3, a4, a5, a6);
  *a1 = v9;
  a1[1] = 0;
  return result;
}
