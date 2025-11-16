// Function: sub_33AC130
// Address: 0x33ac130
//
void __fastcall sub_33AC130(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rdx
  __int64 v4; // rsi
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rax
  int v17; // edx
  __int64 v18; // r12
  int v19; // r13d
  __int128 v20; // [rsp-30h] [rbp-B0h]
  __int128 v21; // [rsp-20h] [rbp-A0h]
  __int64 v22; // [rsp+0h] [rbp-80h]
  __int64 v23; // [rsp+8h] [rbp-78h]
  __int128 v24; // [rsp+10h] [rbp-70h]
  __int64 v25; // [rsp+40h] [rbp-40h] BYREF
  int v26; // [rsp+48h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 864);
  *(_QWORD *)&v24 = sub_33F1270(v2, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  *((_QWORD *)&v24 + 1) = v3;
  v4 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v5 = sub_338B750(a1, v4);
  v7 = v6;
  v25 = 0;
  v12 = sub_33738B0(a1, v4, v6, v8, v9, v10);
  v13 = v11;
  v14 = *(_QWORD *)a1;
  v26 = *(_DWORD *)(a1 + 848);
  if ( v14 )
  {
    if ( &v25 != (__int64 *)(v14 + 48) )
    {
      v15 = *(_QWORD *)(v14 + 48);
      v25 = v15;
      if ( v15 )
      {
        v22 = v12;
        v23 = v11;
        sub_B96E90((__int64)&v25, v15, 1);
        v12 = v22;
        v13 = v23;
      }
    }
  }
  *((_QWORD *)&v21 + 1) = v7;
  *(_QWORD *)&v21 = v5;
  *((_QWORD *)&v20 + 1) = v13;
  *(_QWORD *)&v20 = v12;
  v16 = sub_340F900(v2, 319, (unsigned int)&v25, 1, 0, v13, v20, v21, v24);
  v18 = v16;
  v19 = v17;
  if ( v16 )
  {
    nullsub_1875(v16, v2, 0);
    *(_QWORD *)(v2 + 384) = v18;
    *(_DWORD *)(v2 + 392) = v19;
    sub_33E2B60(v2, 0);
  }
  else
  {
    *(_QWORD *)(v2 + 384) = 0;
    *(_DWORD *)(v2 + 392) = v17;
  }
  if ( v25 )
    sub_B91220((__int64)&v25, v25);
}
