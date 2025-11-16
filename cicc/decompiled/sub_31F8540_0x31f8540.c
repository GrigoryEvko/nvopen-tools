// Function: sub_31F8540
// Address: 0x31f8540
//
__int64 __fastcall sub_31F8540(__int64 a1)
{
  __int64 v2; // rax
  int v3; // r14d
  int v4; // r12d
  int v5; // eax
  __int64 v6; // rax
  __int16 v7; // [rsp+8h] [rbp-58h] BYREF
  int v8; // [rsp+Ah] [rbp-56h]
  __int16 v9; // [rsp+Eh] [rbp-52h]
  _DWORD v10[4]; // [rsp+10h] [rbp-50h] BYREF
  char v11; // [rsp+22h] [rbp-3Eh]

  if ( !*(_DWORD *)(a1 + 1332) )
  {
    v7 = 4097;
    v10[0] = 116;
    v8 = 116;
    v9 = 1;
    v2 = sub_3709240(a1 + 648, &v7);
    v3 = sub_3707F80(a1 + 632, v2);
    v4 = 2 * (sub_AE2980(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 2488LL) + 312LL, 0)[1] >> 3 == 8) + 10;
    v5 = sub_AE2980(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 2488LL) + 312LL, 0)[1];
    LOWORD(v10[0]) = 4098;
    *(_DWORD *)((char *)v10 + 2) = v3;
    v11 = 0;
    v10[2] = v4 | (v5 << 10) & 0x1FE000;
    v6 = sub_3708FB0(a1 + 648, v10);
    *(_DWORD *)(a1 + 1332) = sub_3707F80(a1 + 632, v6);
  }
  return *(unsigned int *)(a1 + 1332);
}
