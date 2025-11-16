// Function: sub_12A1B40
// Address: 0x12a1b40
//
__int64 __fastcall sub_12A1B40(__int64 a1, __int64 a2)
{
  int v3; // esi
  __int64 v4; // rax
  int v5; // edx
  int v6; // ecx
  int v8; // eax
  int v9; // [rsp+0h] [rbp-30h]
  __int64 v10; // [rsp+18h] [rbp-18h] BYREF

  v3 = sub_12A0C10(a1, *(_QWORD *)(a2 + 160));
  v4 = *(_QWORD *)(a2 + 128);
  v5 = 8 * v4;
  if ( *(char *)(a2 + 142) >= 0 && *(_BYTE *)(a2 + 140) == 12 )
  {
    v9 = 8 * v4;
    v8 = sub_8D4AB0(a2);
    v5 = v9;
    v6 = 8 * v8;
  }
  else
  {
    v6 = 8 * *(_DWORD *)(a2 + 136);
  }
  v10 = 0x10000000CLL;
  return sub_15A5AA0((int)a1 + 16, v3, v5, v6, (unsigned int)&v10, 0, (__int64)byte_3F871B3, 0);
}
