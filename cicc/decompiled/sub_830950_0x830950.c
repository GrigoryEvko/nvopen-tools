// Function: sub_830950
// Address: 0x830950
//
__int64 __fastcall sub_830950(__int64 a1, int a2)
{
  __int64 result; // rax
  __int64 v3; // rax
  __int64 v4[3]; // [rsp+8h] [rbp-18h] BYREF

  result = sub_830310(0, v4, a2, 0);
  if ( (_DWORD)result )
  {
    v3 = sub_8D46C0(v4[0]);
    return (unsigned int)sub_8D5DF0(v3) != 0;
  }
  return result;
}
