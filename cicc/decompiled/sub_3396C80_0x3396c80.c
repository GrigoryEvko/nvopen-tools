// Function: sub_3396C80
// Address: 0x3396c80
//
void __fastcall sub_3396C80(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 v6; // rax
  int v7; // eax
  int v8; // edx
  __int64 v9; // r9
  int v10; // r8d
  int v11; // ecx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // r12
  int v15; // edx
  int v16; // r13d
  _QWORD *v17; // rax
  __int64 v18; // rsi
  int v19; // [rsp+8h] [rbp-78h]
  int v20; // [rsp+10h] [rbp-70h]
  __int64 *v21; // [rsp+18h] [rbp-68h]
  int v22; // [rsp+18h] [rbp-68h]
  __int64 v23; // [rsp+38h] [rbp-48h] BYREF
  __int64 v24; // [rsp+40h] [rbp-40h] BYREF
  int v25; // [rsp+48h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(__int64 **)(a2 - 8);
  else
    v3 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  sub_338B750(a1, *v3);
  v4 = *(_QWORD *)(a1 + 864);
  v5 = *(_QWORD *)(v4 + 16);
  v21 = *(__int64 **)(a2 + 8);
  v6 = sub_2E79000(*(__int64 **)(v4 + 40));
  v7 = sub_2D5BAE0(v5, v6, v21, 0);
  v24 = 0;
  v9 = *(_QWORD *)(a1 + 864);
  v10 = v8;
  v11 = v7;
  v12 = *(_QWORD *)a1;
  v25 = *(_DWORD *)(a1 + 848);
  if ( v12 )
  {
    if ( &v24 != (__int64 *)(v12 + 48) )
    {
      v13 = *(_QWORD *)(v12 + 48);
      v24 = v13;
      if ( v13 )
      {
        v19 = v11;
        v20 = v8;
        v22 = v9;
        sub_B96E90((__int64)&v24, v13, 1);
        v11 = v19;
        v10 = v20;
        LODWORD(v9) = v22;
      }
    }
  }
  v23 = a2;
  v14 = sub_33FAF80(v9, 227, (unsigned int)&v24, v11, v10, v9);
  v16 = v15;
  v17 = sub_337DC20(a1 + 8, &v23);
  *v17 = v14;
  v18 = v24;
  *((_DWORD *)v17 + 2) = v16;
  if ( v18 )
    sub_B91220((__int64)&v24, v18);
}
