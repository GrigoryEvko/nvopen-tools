// Function: sub_31CD6C0
// Address: 0x31cd6c0
//
__int64 __fastcall sub_31CD6C0(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4)
{
  __int64 *v4; // r14
  __int64 v5; // r13
  _QWORD *v6; // rax
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // r15
  _QWORD *v14; // r13
  __int64 v18[4]; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v19[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v20; // [rsp+50h] [rbp-40h]

  v4 = (__int64 *)sub_B43CA0(a2);
  v5 = sub_BCB2D0((_QWORD *)*v4);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v6 = *(_QWORD **)(a2 - 8);
  else
    v6 = (_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v19[0] = *(_QWORD *)(*v6 + 8LL);
  v19[1] = *(_QWORD *)(v6[4] + 8LL);
  v7 = sub_B6E160(v4, a3, (__int64)v19, 2);
  v8 = sub_AD64C0(v5, a4, 0);
  v20 = 257;
  v18[0] = v8;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v9 = *(__int64 **)(a2 - 8);
  else
    v9 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v10 = *v9;
  v11 = *(_QWORD *)(a1 + 304);
  v12 = v9[4];
  v18[1] = v10;
  v13 = 0;
  v18[2] = v12;
  v18[3] = *(_QWORD *)(v11 + 32 * (1LL - (*(_DWORD *)(v11 + 4) & 0x7FFFFFF)));
  if ( v7 )
    v13 = *(_QWORD *)(v7 + 24);
  v14 = sub_BD2C40(88, 5u);
  if ( v14 )
  {
    sub_B44260((__int64)v14, **(_QWORD **)(v13 + 16), 56, 5u, a2 + 24, 0);
    v14[9] = 0;
    sub_B4A290((__int64)v14, v13, v7, v18, 4, (__int64)v19, 0, 0);
  }
  sub_BD84D0(a2, (__int64)v14);
  return sub_B43D60((_QWORD *)a2);
}
