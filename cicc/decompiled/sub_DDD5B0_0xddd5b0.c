// Function: sub_DDD5B0
// Address: 0xddd5b0
//
__int64 __fastcall sub_DDD5B0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax

  if ( !a2 )
    return 0;
  result = sub_DCD020(a1, a3, a4, a5);
  if ( !(_BYTE)result )
    return sub_DDC560(a1, **(_QWORD **)(a2 + 32), a3, a4, a5);
  return result;
}
