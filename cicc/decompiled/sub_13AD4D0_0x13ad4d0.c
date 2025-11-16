// Function: sub_13AD4D0
// Address: 0x13ad4d0
//
__int64 __fastcall sub_13AD4D0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 *a4, __int64 a5)
{
  __int64 result; // rax

  *(_BYTE *)(a5 + 43) = 0;
  result = sub_13AB330(a1, a2, a3, a5);
  if ( !(_BYTE)result )
    return sub_13AD2A0(a1, a2, a3, a4, a5);
  return result;
}
