// Function: sub_1E09220
// Address: 0x1e09220
//
__int64 __fastcall sub_1E09220(__int64 a1, unsigned int a2, __int64 a3)
{
  bool v3; // zf
  unsigned int v4; // r12d
  __m128i v6; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v7; // [rsp+10h] [rbp-30h]
  int v8; // [rsp+14h] [rbp-2Ch]
  __int64 v9; // [rsp+18h] [rbp-28h]
  int v10; // [rsp+20h] [rbp-20h]
  char v11; // [rsp+24h] [rbp-1Ch]

  v3 = *(_BYTE *)(a1 + 4) == 0;
  v4 = *(_DWORD *)a1;
  *(_BYTE *)(a1 + 36) = 1;
  if ( !v3 || v4 >= a2 )
    v4 = a2;
  v9 = a3;
  v7 = v4;
  v6 = 0u;
  v8 = 0;
  v10 = 256;
  v11 = 0;
  sub_1E090A0(a1 + 8, &v6);
  sub_1E08740(a1, v4);
  return ~*(_DWORD *)(a1 + 32) - 858993459 * (unsigned int)((__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 3);
}
