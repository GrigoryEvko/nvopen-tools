// Function: sub_2472770
// Address: 0x2472770
//
void __fastcall sub_2472770(__int64 a1, __int64 a2)
{
  int v2; // edx
  __int64 v3; // rdx
  __int64 **v4; // r15
  unsigned int v5; // ebx
  unsigned int v6; // eax
  unsigned __int64 v7; // [rsp+18h] [rbp-158h]
  int v8; // [rsp+28h] [rbp-148h]
  _QWORD v9[4]; // [rsp+30h] [rbp-140h] BYREF
  _BYTE v10[32]; // [rsp+50h] [rbp-120h] BYREF
  __int16 v11; // [rsp+70h] [rbp-100h]
  _BYTE v12[32]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v13; // [rsp+A0h] [rbp-D0h]
  unsigned int *v14[2]; // [rsp+B0h] [rbp-C0h] BYREF
  char v15; // [rsp+C0h] [rbp-B0h] BYREF
  void *v16; // [rsp+130h] [rbp-40h]

  sub_246F3F0(a1, *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
  sub_23D0AB0((__int64)v14, a2, 0, 0, 0);
  v2 = *(_DWORD *)(a2 + 4);
  v13 = 257;
  v3 = v2 & 0x7FFFFFF;
  v11 = 257;
  v9[0] = *(_QWORD *)(a2 - 32 * v3);
  v9[1] = *(_QWORD *)(a2 + 32 * (1 - v3));
  v4 = *(__int64 ***)(*(_QWORD *)(a1 + 8) + 80LL);
  v7 = *(_QWORD *)(a2 + 32 * (2 - v3));
  v5 = sub_BCB060(*(_QWORD *)(v7 + 8));
  v6 = sub_BCB060((__int64)v4);
  v9[2] = sub_24633A0((__int64 *)v14, (unsigned int)(v5 <= v6) + 38, v7, v4, (__int64)v10, 0, v8, 0);
  sub_921880(
    v14,
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 408LL),
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 416LL),
    (int)v9,
    3,
    (__int64)v12,
    0);
  sub_B43D60((_QWORD *)a2);
  nullsub_61();
  v16 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v14[0] != &v15 )
    _libc_free((unsigned __int64)v14[0]);
}
