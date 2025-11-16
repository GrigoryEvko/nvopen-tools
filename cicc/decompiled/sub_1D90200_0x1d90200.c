// Function: sub_1D90200
// Address: 0x1d90200
//
__int64 __fastcall sub_1D90200(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // r14
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  _DWORD v14[4]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v15; // [rsp+20h] [rbp-50h]
  __int64 v16; // [rsp+28h] [rbp-48h]
  int v17; // [rsp+30h] [rbp-40h]

  v6 = sub_38BFA60(*(_QWORD *)(*(_QWORD *)(a2 + 56) + 24LL), 1);
  v7 = *(_QWORD *)(a2 + 56);
  v8 = v6;
  v9 = sub_1E0B640(v7, *(_QWORD *)(*(_QWORD *)(a1 + 248) + 8LL) + 256LL, a4, 0, a1);
  sub_1DD5BA0(a2 + 16, v9);
  v10 = *(_QWORD *)v9;
  v11 = *a3;
  *(_QWORD *)(v9 + 8) = a3;
  v11 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v9 = v11 | v10 & 7;
  *(_QWORD *)(v11 + 8) = v9;
  v12 = *a3;
  LOBYTE(v14[0]) = 15;
  v16 = v8;
  v14[0] &= 0xFFF000FF;
  *a3 = v9 | v12 & 7;
  v15 = 0;
  v14[2] = 0;
  v17 = 0;
  sub_1E1A9C0(v9, v7, v14);
  return v8;
}
