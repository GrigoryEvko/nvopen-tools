// Function: sub_2E77B40
// Address: 0x2e77b40
//
__int64 __fastcall sub_2E77B40(__int64 a1, unsigned __int8 a2, __int64 a3)
{
  unsigned __int8 v3; // r12
  bool v4; // zf
  unsigned __int8 v5; // al
  __m128i v7; // [rsp+0h] [rbp-40h] BYREF
  unsigned __int8 v8; // [rsp+10h] [rbp-30h]
  int v9; // [rsp+11h] [rbp-2Fh]
  __int64 v10; // [rsp+18h] [rbp-28h]
  int v11; // [rsp+20h] [rbp-20h]
  char v12; // [rsp+24h] [rbp-1Ch]

  v3 = a2;
  v4 = *(_BYTE *)(a1 + 1) == 0;
  v5 = *(_BYTE *)a1;
  *(_BYTE *)(a1 + 36) = 1;
  if ( v4 && a2 > v5 )
    v3 = v5;
  v10 = a3;
  v8 = v3;
  v7 = 0u;
  v9 = 0;
  v11 = 256;
  v12 = 0;
  sub_2E77AF0((unsigned __int64 *)(a1 + 8), &v7);
  sub_2E76F70(a1, v3);
  return ~*(_DWORD *)(a1 + 32) - 858993459 * (unsigned int)((__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 3);
}
