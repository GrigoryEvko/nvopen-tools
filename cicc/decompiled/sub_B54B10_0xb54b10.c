// Function: sub_B54B10
// Address: 0xb54b10
//
__int64 __fastcall sub_B54B10(unsigned __int8 *a1)
{
  __int64 v1; // rdx
  __int64 v2; // rsi
  int v3; // edi
  _BYTE v5[32]; // [rsp+0h] [rbp-30h] BYREF
  __int16 v6; // [rsp+20h] [rbp-10h]

  v1 = *((_QWORD *)a1 - 4);
  v2 = *((_QWORD *)a1 - 8);
  v3 = *a1;
  v6 = 257;
  return sub_B504D0(v3 - 29, v2, v1, (__int64)v5, 0, 0);
}
