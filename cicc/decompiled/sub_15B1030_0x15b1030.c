// Function: sub_15B1030
// Address: 0x15b1030
//
__int64 __fastcall sub_15B1030(__int64 a1)
{
  __int64 result; // rax

  for ( result = a1; *(_BYTE *)result == 19; result = *(_QWORD *)(result + 8 * (1LL - *(unsigned int *)(result + 8))) )
    ;
  return result;
}
