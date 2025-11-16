// Function: sub_B54AE0
// Address: 0xb54ae0
//
__int64 __fastcall sub_B54AE0(unsigned __int8 *a1)
{
  __int64 v1; // rsi
  int v2; // edi
  _BYTE v4[32]; // [rsp+0h] [rbp-30h] BYREF
  __int16 v5; // [rsp+20h] [rbp-10h]

  v1 = *((_QWORD *)a1 - 4);
  v2 = *a1;
  v5 = 257;
  return sub_B50340(v2 - 29, v1, (__int64)v4, 0, 0);
}
