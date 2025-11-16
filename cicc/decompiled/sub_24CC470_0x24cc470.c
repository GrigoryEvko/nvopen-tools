// Function: sub_24CC470
// Address: 0x24cc470
//
__int64 __fastcall sub_24CC470(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 v5; // rbx
  __int64 v7; // [rsp+8h] [rbp-48h] BYREF
  char v8; // [rsp+10h] [rbp-40h] BYREF

  v4 = a1 + 32;
  v5 = a1 + 80;
  if ( (unsigned __int8)sub_29F38C0(a3, "nosanitize_thread", 17) )
  {
    *(_QWORD *)(a1 + 8) = v4;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 56) = v5;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  else
  {
    v7 = a3;
    sub_2A41C90(
      (unsigned int)&v8,
      a3,
      (unsigned int)"tsan.module_ctor",
      16,
      (unsigned int)"__tsan_init",
      11,
      0,
      0,
      0,
      0,
      (__int64)sub_24CC1B0,
      (__int64)&v7,
      0,
      0,
      0);
    memset((void *)a1, 0, 0x60u);
    *(_QWORD *)(a1 + 8) = v4;
    *(_QWORD *)(a1 + 56) = v5;
    *(_DWORD *)(a1 + 16) = 2;
    *(_BYTE *)(a1 + 28) = 1;
    *(_DWORD *)(a1 + 64) = 2;
    *(_BYTE *)(a1 + 76) = 1;
    return a1;
  }
}
