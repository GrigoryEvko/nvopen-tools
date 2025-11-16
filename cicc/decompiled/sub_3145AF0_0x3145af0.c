// Function: sub_3145AF0
// Address: 0x3145af0
//
__int64 __fastcall sub_3145AF0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2[2]; // [rsp+8h] [rbp-18h] BYREF

  v2[0] = a1;
  result = sub_A721E0(v2, "statepoint-id", 13);
  if ( !(_BYTE)result )
    return sub_A721E0(v2, "statepoint-num-patch-bytes", 26);
  return result;
}
