// Function: sub_24CC3F0
// Address: 0x24cc3f0
//
__int64 __fastcall sub_24CC3F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  unsigned int v4; // eax
  bool v7; // cc
  __int64 result; // rax
  _QWORD v9[4]; // [rsp+0h] [rbp-20h] BYREF

  if ( sub_BCEA30(a1) )
    return 0xFFFFFFFFLL;
  v2 = sub_9208B0(a2, a1);
  v9[1] = v3;
  v9[0] = (v2 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v4 = sub_CA1930(v9);
  if ( ((v4 - 8) & 0xFFFFFFF7) != 0 && ((v4 - 32) & 0xFFFFFFDF) != 0 && v4 != 128 )
    return 0xFFFFFFFFLL;
  _EDX = v4 >> 3;
  __asm { tzcnt   edx, edx }
  v7 = v4 <= 7;
  result = 32;
  if ( !v7 )
    return _EDX;
  return result;
}
