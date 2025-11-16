// Function: sub_D007E0
// Address: 0xd007e0
//
__int64 __fastcall sub_D007E0(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5, unsigned __int8 a6)
{
  __int64 result; // rax
  unsigned int v8; // ebx
  _QWORD v9[3]; // [rsp+8h] [rbp-18h] BYREF

  result = sub_D62CA0(a1, v9, a4, a5, a6 << 16, 0);
  if ( (_BYTE)result )
  {
    v8 = a3 ^ 1;
    result = 0;
    if ( a2 == v9[0] )
      return v8;
  }
  return result;
}
