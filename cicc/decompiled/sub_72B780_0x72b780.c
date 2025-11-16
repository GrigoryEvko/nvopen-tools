// Function: sub_72B780
// Address: 0x72b780
//
__int64 **__fastcall sub_72B780(__int64 a1)
{
  __int64 **result; // rax

  for ( result = *(__int64 ***)(a1 + 112); ((_BYTE)result[3] & 2) == 0; result = (__int64 **)*result )
    ;
  return result;
}
