// Function: sub_6E3F00
// Address: 0x6e3f00
//
__int64 __fastcall sub_6E3F00(int a1, __int64 a2, int a3)
{
  int v3; // ecx
  int v4; // r8d
  __int64 v5; // rdx
  __int64 result; // rax
  int v7; // [rsp+14h] [rbp-14h] BYREF

  v3 = a3 + 68;
  v4 = *(_DWORD *)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 32);
  v7 = 0;
  result = sub_8A2270(a1, *(_QWORD *)(a2 + 24), v5, v3, v4 & 0x4140 | 4u, (unsigned int)&v7, *(_QWORD *)(a2 + 48));
  if ( v7 )
    *(_BYTE *)(a2 + 56) = 1;
  return result;
}
