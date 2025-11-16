// Function: sub_11E6620
// Address: 0x11e6620
//
__int64 __fastcall sub_11E6620(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // eax
  char v5; // r8
  __int64 result; // rax
  char *v7[6]; // [rsp+0h] [rbp-30h] BYREF

  v4 = *(_DWORD *)(a2 + 4);
  v7[0] = 0;
  v7[1] = 0;
  v5 = sub_98B0F0(*(_QWORD *)(a2 - 32LL * (v4 & 0x7FFFFFF)), v7, 1u);
  result = 0;
  if ( v5 )
    return sub_11DBE40(a2, v7, 0, 0xAu, 1, a3);
  return result;
}
