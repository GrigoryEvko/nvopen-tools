// Function: sub_1DB3430
// Address: 0x1db3430
//
__int64 __fastcall sub_1DB3430(int *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  int v6; // eax
  _QWORD v8[2]; // [rsp+0h] [rbp-20h] BYREF
  int v9; // [rsp+10h] [rbp-10h]

  v6 = *a1;
  v8[1] = "%08X";
  v9 = v6;
  v8[0] = &unk_49EFAC8;
  return sub_16E8450(a2, (__int64)v8, (__int64)&unk_49EFAC8, (__int64)"%08X", (int)a1, a6);
}
