// Function: sub_11F1630
// Address: 0x11f1630
//
__int64 __fastcall sub_11F1630(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  int v4[3]; // [rsp+1Ch] [rbp-14h] BYREF

  result = sub_11F0480(a1, a2, a3, 8u, 0);
  if ( !result )
  {
    v4[0] = 0;
    sub_11DA4B0(a2, v4, 1);
    return 0;
  }
  return result;
}
