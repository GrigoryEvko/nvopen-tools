// Function: sub_1495970
// Address: 0x1495970
//
__int64 *__fastcall sub_1495970(__int64 a1, __int64 a2)
{
  __int64 *result; // rax

  result = sub_1473850(a1, a2);
  if ( (result[5] & 4) == 0 )
    return sub_1495470(a1, a2);
  return result;
}
