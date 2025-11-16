// Function: sub_2468000
// Address: 0x2468000
//
void __fastcall sub_2468000(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 **v3; // r14
  unsigned int v4; // r15d
  unsigned int v5; // eax
  unsigned __int64 v6; // rax
  int v7; // edx
  __int64 v8; // rax
  __int64 **v9; // r14
  unsigned int v10; // r15d
  unsigned int v11; // eax
  unsigned __int64 v12; // [rsp+18h] [rbp-188h]
  unsigned __int64 v13; // [rsp+18h] [rbp-188h]
  int v14; // [rsp+28h] [rbp-178h]
  _QWORD v15[4]; // [rsp+30h] [rbp-170h] BYREF
  _BYTE v16[32]; // [rsp+50h] [rbp-150h] BYREF
  __int16 v17; // [rsp+70h] [rbp-130h]
  int v18[8]; // [rsp+80h] [rbp-120h] BYREF
  __int16 v19; // [rsp+A0h] [rbp-100h]
  _BYTE v20[32]; // [rsp+B0h] [rbp-F0h] BYREF
  __int16 v21; // [rsp+D0h] [rbp-D0h]
  unsigned int *v22[2]; // [rsp+E0h] [rbp-C0h] BYREF
  char v23; // [rsp+F0h] [rbp-B0h] BYREF
  _QWORD *v24; // [rsp+128h] [rbp-78h]
  void *v25; // [rsp+160h] [rbp-40h]

  sub_23D0AB0((__int64)v22, a2, 0, 0, 0);
  v21 = 257;
  v2 = *(_DWORD *)(a2 + 4);
  v17 = 257;
  v15[0] = *(_QWORD *)(a2 - 32LL * (v2 & 0x7FFFFFF));
  v3 = (__int64 **)sub_BCB2D0(v24);
  v12 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v4 = sub_BCB060(*(_QWORD *)(v12 + 8));
  v5 = sub_BCB060((__int64)v3);
  v6 = sub_24633A0((__int64 *)v22, (unsigned int)(v4 <= v5) + 38, v12, v3, (__int64)v16, 0, v18[0], 0);
  v7 = *(_DWORD *)(a2 + 4);
  v15[1] = v6;
  v8 = *(_QWORD *)(a1 + 8);
  v19 = 257;
  v9 = *(__int64 ***)(v8 + 80);
  v13 = *(_QWORD *)(a2 + 32 * (2LL - (v7 & 0x7FFFFFF)));
  v10 = sub_BCB060(*(_QWORD *)(v13 + 8));
  v11 = sub_BCB060((__int64)v9);
  v15[2] = sub_24633A0((__int64 *)v22, (unsigned int)(v10 <= v11) + 38, v13, v9, (__int64)v18, 0, v14, 0);
  sub_921880(
    v22,
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 424LL),
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 432LL),
    (int)v15,
    3,
    (__int64)v20,
    0);
  sub_B43D60((_QWORD *)a2);
  nullsub_61();
  v25 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v22[0] != &v23 )
    _libc_free((unsigned __int64)v22[0]);
}
