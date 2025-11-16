// Function: sub_33AC2B0
// Address: 0x33ac2b0
//
void __fastcall sub_33AC2B0(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // rdx
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  int v15; // r9d
  __int64 v16; // r10
  __int64 v17; // r11
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rax
  int v21; // edx
  __int64 v22; // r12
  int v23; // r13d
  __int128 v24; // [rsp-50h] [rbp-F0h]
  __int128 v25; // [rsp-20h] [rbp-C0h]
  __int64 v26; // [rsp+0h] [rbp-A0h]
  __int64 v27; // [rsp+8h] [rbp-98h]
  __int128 v28; // [rsp+10h] [rbp-90h]
  __int128 v29; // [rsp+20h] [rbp-80h]
  __int128 v30; // [rsp+30h] [rbp-70h]
  __int64 v31; // [rsp+60h] [rbp-40h] BYREF
  int v32; // [rsp+68h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 864);
  *(_QWORD *)&v28 = sub_33F1270(v3, *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
  *((_QWORD *)&v28 + 1) = v4;
  v5 = sub_33F1270(*(_QWORD *)(a1 + 864), *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v7 = v6;
  *(_QWORD *)&v29 = sub_338B750(a1, *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
  *((_QWORD *)&v29 + 1) = v8;
  v9 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  *(_QWORD *)&v30 = sub_338B750(a1, v9);
  *((_QWORD *)&v30 + 1) = v10;
  v31 = 0;
  v16 = sub_33738B0(a1, v9, v10, v11, v12, v13);
  v17 = v14;
  v18 = *(_QWORD *)a1;
  v32 = *(_DWORD *)(a1 + 848);
  if ( v18 )
  {
    if ( &v31 != (__int64 *)(v18 + 48) )
    {
      v19 = *(_QWORD *)(v18 + 48);
      v31 = v19;
      if ( v19 )
      {
        v26 = v16;
        v27 = v14;
        sub_B96E90((__int64)&v31, v19, 1);
        v16 = v26;
        v17 = v27;
      }
    }
  }
  *((_QWORD *)&v25 + 1) = v7;
  *(_QWORD *)&v25 = v5;
  *((_QWORD *)&v24 + 1) = v17;
  *(_QWORD *)&v24 = v16;
  v20 = sub_33FC1D0(v3, 318, (unsigned int)&v31, 1, 0, v15, v24, v30, v29, v25, v28);
  v22 = v20;
  v23 = v21;
  if ( v20 )
  {
    nullsub_1875(v20, v3, 0);
    *(_QWORD *)(v3 + 384) = v22;
    *(_DWORD *)(v3 + 392) = v23;
    sub_33E2B60(v3, 0);
  }
  else
  {
    *(_QWORD *)(v3 + 384) = 0;
    *(_DWORD *)(v3 + 392) = v21;
  }
  if ( v31 )
    sub_B91220((__int64)&v31, v31);
}
