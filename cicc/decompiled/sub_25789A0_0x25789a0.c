// Function: sub_25789A0
// Address: 0x25789a0
//
__int64 __fastcall sub_25789A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  __int64 v9; // [rsp+8h] [rbp-18h] BYREF

  v7 = *(_QWORD *)(a1 + 8);
  v9 = a2;
  **(_BYTE **)a1 |= sub_25784B0(v7, &v9, a3, a4, a5, a6);
  return 1;
}
