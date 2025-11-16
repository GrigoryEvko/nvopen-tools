// Function: sub_989FD0
// Address: 0x989fd0
//
__int64 __fastcall sub_989FD0(unsigned int a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 result; // rax
  _DWORD v6[2]; // [rsp+0h] [rbp-20h] BYREF
  __int64 v7; // [rsp+8h] [rbp-18h]

  sub_989E10((__int64)v6, a1, a2, a3, a4, a5);
  result = v7;
  if ( !v7 || v6[1] != (~LOWORD(v6[0]) & 0x3FF) )
    return 0;
  return result;
}
