// Function: sub_11CA290
// Address: 0x11ca290
//
__int64 __fastcall sub_11CA290(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 *v5; // rsi
  _QWORD v7[2]; // [rsp+8h] [rbp-20h] BYREF
  _QWORD v8[2]; // [rsp+18h] [rbp-10h] BYREF

  v5 = *(__int64 **)(a1 + 8);
  v8[0] = a1;
  v8[1] = a2;
  v7[0] = v5;
  v7[1] = v5;
  return sub_11C9AF0(0x1CFu, v5, v7, 2, (int)v8, 2, a3, a4, 0);
}
