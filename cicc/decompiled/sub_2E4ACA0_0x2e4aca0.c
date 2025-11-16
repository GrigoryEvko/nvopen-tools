// Function: sub_2E4ACA0
// Address: 0x2e4aca0
//
unsigned __int64 __fastcall sub_2E4ACA0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rdx
  __int64 v5; // rsi
  unsigned int v6; // eax
  __int64 v7; // rdi
  __int16 v8; // r12
  __int64 v9; // rdx
  __int16 *v10; // r13
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  _QWORD v19[4]; // [rsp+0h] [rbp-80h] BYREF
  __int64 v20; // [rsp+20h] [rbp-60h]
  __int16 *v21; // [rsp+28h] [rbp-58h]
  __int64 v22; // [rsp+30h] [rbp-50h]
  __int64 v23; // [rsp+40h] [rbp-40h]
  __int16 *v24; // [rsp+48h] [rbp-38h]
  __int64 v25; // [rsp+50h] [rbp-30h]

  sub_2E44C10((__int64)v19, a2, *(_QWORD *)a1, **(_BYTE **)(a1 + 8));
  v3 = *(_QWORD *)(a1 + 16);
  v4 = *(_QWORD *)(v3 + 8);
  v5 = *(_QWORD *)(v3 + 56);
  v6 = *(_DWORD *)(v4 + 24LL * *(unsigned int *)(v19[0] + 8LL) + 16);
  LODWORD(v4) = *(_DWORD *)(v4 + 24LL * *(unsigned int *)(v19[1] + 8LL) + 16);
  v7 = *(_QWORD *)(a1 + 24);
  v8 = v4;
  v9 = (unsigned int)v4 >> 12;
  LODWORD(v20) = v6 & 0xFFF;
  v10 = (__int16 *)(v5 + 2 * v9);
  v21 = (__int16 *)(v5 + 2LL * (v6 >> 12));
  LODWORD(v22) = v20;
  sub_2E4AB00(v7, 0, v9, v6 & 0xFFF, v11, v12, v20, v21, v20);
  LODWORD(v23) = v8 & 0xFFF;
  v24 = v10;
  v13 = *(_QWORD *)(a1 + 24);
  LODWORD(v25) = v23;
  return sub_2E4AB00(v13, 0, v14, v15, v16, v17, v23, v10, v23);
}
