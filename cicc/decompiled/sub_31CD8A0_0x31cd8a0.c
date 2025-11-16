// Function: sub_31CD8A0
// Address: 0x31cd8a0
//
__int64 __fastcall sub_31CD8A0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 *v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r15
  _QWORD *v12; // r13
  __int64 v14[4]; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v15[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v16; // [rsp+50h] [rbp-40h]

  v5 = (__int64 *)sub_B43CA0(a2);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v6 = *(_QWORD *)(a2 - 8);
  else
    v6 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v15[0] = *(_QWORD *)(*(_QWORD *)(v6 + 32) + 8LL);
  v15[1] = *(_QWORD *)(*(_QWORD *)(v6 + 64) + 8LL);
  v7 = sub_B6E160(v5, a3, (__int64)v15, 2);
  v16 = 257;
  v8 = v7;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v9 = *(__int64 **)(a2 - 8);
  else
    v9 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v14[0] = *v9;
  v14[1] = v9[4];
  v10 = *(_QWORD *)(a1 + 304);
  v11 = 0;
  v14[2] = v9[8];
  v14[3] = *(_QWORD *)(v10 + 32 * (1LL - (*(_DWORD *)(v10 + 4) & 0x7FFFFFF)));
  if ( v8 )
    v11 = *(_QWORD *)(v8 + 24);
  v12 = sub_BD2C40(88, 5u);
  if ( v12 )
  {
    sub_B44260((__int64)v12, **(_QWORD **)(v11 + 16), 56, 5u, a2 + 24, 0);
    v12[9] = 0;
    sub_B4A290((__int64)v12, v11, v8, v14, 4, (__int64)v15, 0, 0);
  }
  sub_BD84D0(a2, (__int64)v12);
  return sub_B43D60((_QWORD *)a2);
}
