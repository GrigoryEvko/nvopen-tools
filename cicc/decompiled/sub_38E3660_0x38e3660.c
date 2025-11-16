// Function: sub_38E3660
// Address: 0x38e3660
//
__int64 __fastcall sub_38E3660(__int64 a1, unsigned __int64 a2, __int64 a3, unsigned __int64 a4, unsigned __int64 a5)
{
  __int64 *v6; // rdi
  unsigned __int64 v8[3]; // [rsp+8h] [rbp-20h] BYREF

  *(_BYTE *)(a1 + 17) = 1;
  v6 = *(__int64 **)(a1 + 344);
  v8[0] = a4;
  v8[1] = a5;
  sub_16D14E0(v6, a2, 0, a3, v8, 1, 0, 0, 1u);
  sub_38E35B0((_QWORD *)a1);
  return 1;
}
