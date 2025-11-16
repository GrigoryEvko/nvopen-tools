// Function: sub_1047A80
// Address: 0x1047a80
//
void __fastcall sub_1047A80(__int64 a1)
{
  __int64 v2; // rax
  __int64 *v3; // rax
  __int64 *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rsi
  _QWORD v9[4]; // [rsp+10h] [rbp-C80h] BYREF
  __int64 v10[4]; // [rsp+30h] [rbp-C60h] BYREF
  _QWORD v11[3]; // [rsp+50h] [rbp-C40h] BYREF
  __int64 v12; // [rsp+68h] [rbp-C28h]
  __int64 v13; // [rsp+70h] [rbp-C20h] BYREF
  unsigned int v14; // [rsp+78h] [rbp-C18h]
  _QWORD v15[2]; // [rsp+1B0h] [rbp-AE0h] BYREF
  char v16; // [rsp+1C0h] [rbp-AD0h]
  _BYTE *v17; // [rsp+1C8h] [rbp-AC8h]
  __int64 v18; // [rsp+1D0h] [rbp-AC0h]
  _BYTE v19[128]; // [rsp+1D8h] [rbp-AB8h] BYREF
  __int16 v20; // [rsp+258h] [rbp-A38h]
  _QWORD v21[2]; // [rsp+260h] [rbp-A30h] BYREF
  __int64 v22; // [rsp+270h] [rbp-A20h]
  __int64 v23; // [rsp+278h] [rbp-A18h] BYREF
  unsigned int v24; // [rsp+280h] [rbp-A10h]
  char v25; // [rsp+2F8h] [rbp-998h] BYREF
  _QWORD v26[5]; // [rsp+300h] [rbp-990h] BYREF
  _BYTE *v27; // [rsp+328h] [rbp-968h]
  __int64 v28; // [rsp+330h] [rbp-960h]
  _BYTE v29[2304]; // [rsp+338h] [rbp-958h] BYREF
  __int64 v30; // [rsp+C38h] [rbp-58h]
  __int64 v31; // [rsp+C40h] [rbp-50h]
  __int64 v32; // [rsp+C48h] [rbp-48h]
  __int64 v33; // [rsp+C50h] [rbp-40h]
  __int64 v34; // [rsp+C58h] [rbp-38h]

  v2 = *(_QWORD *)a1;
  v11[2] = 0;
  v12 = 1;
  v11[0] = v2;
  v11[1] = v2;
  v3 = &v13;
  do
  {
    *v3 = -4;
    v3 += 5;
    *(v3 - 4) = -3;
    *(v3 - 3) = -4;
    *(v3 - 2) = -3;
  }
  while ( v3 != v15 );
  v15[1] = 0;
  v18 = 0x400000000LL;
  v20 = 256;
  v15[0] = v21;
  v16 = 0;
  v17 = v19;
  v21[1] = 0;
  v22 = 1;
  v21[0] = &unk_49DDBE8;
  v4 = &v23;
  do
  {
    *v4 = -4096;
    v4 += 2;
  }
  while ( v4 != (__int64 *)&v25 );
  v26[1] = *(_QWORD *)(a1 + 8);
  v28 = 0x2000000000LL;
  v26[0] = a1;
  v27 = v29;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = a1;
  sub_103DDC0(v9, a1);
  v10[2] = (__int64)v11;
  v9[2] = v26;
  v5 = *(_QWORD *)(a1 + 8);
  v9[0] = &unk_49E5AA8;
  v10[3] = v5;
  v10[0] = a1;
  v10[1] = (__int64)v9;
  sub_10468A0(v10);
  v6 = (unsigned int)v33;
  v7 = v31;
  *(_BYTE *)(a1 + 356) = 1;
  v8 = 56 * v6;
  sub_C7D6A0(v7, 56 * v6, 8);
  if ( v27 != v29 )
    _libc_free(v27, v8);
  v21[0] = &unk_49DDBE8;
  if ( (v22 & 1) == 0 )
  {
    v8 = 16LL * v24;
    sub_C7D6A0(v23, v8, 8);
  }
  nullsub_184();
  if ( v17 != v19 )
    _libc_free(v17, v8);
  if ( (v12 & 1) == 0 )
    sub_C7D6A0(v13, 40LL * v14, 8);
}
