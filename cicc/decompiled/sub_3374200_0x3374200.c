// Function: sub_3374200
// Address: 0x3374200
//
__int64 __fastcall sub_3374200(__int64 *a1)
{
  int v1; // ecx
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // r10
  __int64 v5; // rsi
  int v6; // r9d
  unsigned int v7; // r15d
  __int64 v8; // rax
  __int64 v9; // r12
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rsi
  unsigned int v14; // edx
  int v15; // [rsp+8h] [rbp-C8h]
  __int64 v16; // [rsp+8h] [rbp-C8h]
  __int64 v17; // [rsp+8h] [rbp-C8h]
  __int64 v18; // [rsp+8h] [rbp-C8h]
  __int64 v19; // [rsp+40h] [rbp-90h] BYREF
  int v20; // [rsp+48h] [rbp-88h]
  __int64 v21; // [rsp+50h] [rbp-80h] BYREF
  int v22; // [rsp+58h] [rbp-78h]
  __int64 v23; // [rsp+60h] [rbp-70h]
  __int64 v24; // [rsp+68h] [rbp-68h]
  __int64 v25; // [rsp+70h] [rbp-60h]
  __int64 v26; // [rsp+78h] [rbp-58h]
  __int64 v27; // [rsp+80h] [rbp-50h]
  __int64 v28; // [rsp+88h] [rbp-48h]
  __int64 v29; // [rsp+90h] [rbp-40h]

  v1 = *((_DWORD *)a1 + 212);
  v2 = a1[108];
  v25 = 0;
  v3 = *a1;
  v26 = 0;
  v27 = 0;
  v4 = *(_QWORD *)(v2 + 16);
  v28 = 0;
  LOBYTE(v29) = 0;
  v19 = 0;
  v20 = v1;
  if ( v3 )
  {
    if ( &v19 != (__int64 *)(v3 + 48) )
    {
      v5 = *(_QWORD *)(v3 + 48);
      v19 = v5;
      if ( v5 )
      {
        v15 = v4;
        sub_B96E90((__int64)&v19, v5, 1);
        v2 = a1[108];
        LODWORD(v4) = v15;
      }
    }
  }
  sub_3494590((unsigned int)&v21, v4, v2, 724, 263, 0, 0, 0, v25, v26, v27, v28, v29, (__int64)&v19, 0, 0);
  v7 = v24;
  v8 = v23;
  if ( v19 )
  {
    v16 = v23;
    sub_B91220((__int64)&v19, v19);
    v8 = v16;
  }
  v9 = a1[108];
  if ( (*(_BYTE *)(*(_QWORD *)v9 + 877LL) & 6) == 2 )
  {
    v11 = *((_DWORD *)a1 + 212);
    v12 = *a1;
    v21 = 0;
    v22 = v11;
    if ( v12 )
    {
      if ( &v21 != (__int64 *)(v12 + 48) )
      {
        v13 = *(_QWORD *)(v12 + 48);
        v21 = v13;
        if ( v13 )
          sub_B96E90((__int64)&v21, v13, 1);
      }
    }
    v8 = sub_33FAF80(v9, 331, (unsigned int)&v21, 1, 0, v6);
    v7 = v14;
    if ( v21 )
    {
      v18 = v8;
      sub_B91220((__int64)&v21, v21);
      v8 = v18;
    }
    v9 = a1[108];
  }
  if ( v8 )
  {
    v17 = v8;
    nullsub_1875(v8, v9, 0);
    *(_QWORD *)(v9 + 384) = v17;
    *(_DWORD *)(v9 + 392) = v7;
    return sub_33E2B60(v9, 0);
  }
  else
  {
    *(_QWORD *)(v9 + 384) = 0;
    *(_DWORD *)(v9 + 392) = v7;
    return v7;
  }
}
