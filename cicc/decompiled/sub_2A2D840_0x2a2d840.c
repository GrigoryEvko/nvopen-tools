// Function: sub_2A2D840
// Address: 0x2a2d840
//
__int64 __fastcall sub_2A2D840(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rsi
  __int64 v4; // r9
  __int64 v5; // r8
  unsigned int *v6; // rax
  int v7; // ecx
  unsigned int *v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 **v11; // rdi
  _BYTE *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v17; // rsi
  _BYTE *v18; // [rsp+0h] [rbp-120h]
  _BYTE *v19; // [rsp+8h] [rbp-118h]
  __int64 v20; // [rsp+10h] [rbp-110h]
  int v21; // [rsp+2Ch] [rbp-F4h] BYREF
  __int64 v22[4]; // [rsp+30h] [rbp-F0h] BYREF
  __int16 v23; // [rsp+50h] [rbp-D0h]
  unsigned int *v24; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v25; // [rsp+68h] [rbp-B8h]
  _BYTE v26[32]; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v27; // [rsp+90h] [rbp-90h]
  __int64 v28; // [rsp+98h] [rbp-88h]
  __int16 v29; // [rsp+A0h] [rbp-80h]
  __int64 v30; // [rsp+A8h] [rbp-78h]
  void **v31; // [rsp+B0h] [rbp-70h]
  void **v32; // [rsp+B8h] [rbp-68h]
  __int64 v33; // [rsp+C0h] [rbp-60h]
  int v34; // [rsp+C8h] [rbp-58h]
  __int16 v35; // [rsp+CCh] [rbp-54h]
  char v36; // [rsp+CEh] [rbp-52h]
  __int64 v37; // [rsp+D0h] [rbp-50h]
  __int64 v38; // [rsp+D8h] [rbp-48h]
  void *v39; // [rsp+E0h] [rbp-40h] BYREF
  void *v40; // [rsp+E8h] [rbp-38h] BYREF

  v31 = &v39;
  v30 = sub_BD5C60(a1);
  v24 = (unsigned int *)v26;
  v39 = &unk_49DA100;
  v25 = 0x200000000LL;
  v32 = &v40;
  v40 = &unk_49DA0B0;
  v2 = *(_QWORD *)(a1 + 40);
  v33 = 0;
  v27 = v2;
  v34 = 0;
  v35 = 512;
  v36 = 7;
  v37 = 0;
  v38 = 0;
  v28 = a1 + 24;
  v29 = 0;
  v3 = *(_QWORD *)sub_B46C60(a1);
  v22[0] = v3;
  if ( v3 && (sub_B96E90((__int64)v22, v3, 1), (v5 = v22[0]) != 0) )
  {
    v6 = v24;
    v7 = v25;
    v8 = &v24[4 * (unsigned int)v25];
    if ( v24 != v8 )
    {
      while ( *v6 )
      {
        v6 += 4;
        if ( v8 == v6 )
          goto LABEL_15;
      }
      *((_QWORD *)v6 + 1) = v22[0];
      goto LABEL_8;
    }
LABEL_15:
    if ( (unsigned int)v25 >= (unsigned __int64)HIDWORD(v25) )
    {
      v17 = (unsigned int)v25 + 1LL;
      if ( HIDWORD(v25) < v17 )
      {
        v20 = v22[0];
        sub_C8D5F0((__int64)&v24, v26, v17, 0x10u, v22[0], v4);
        v5 = v20;
        v8 = &v24[4 * (unsigned int)v25];
      }
      *(_QWORD *)v8 = 0;
      *((_QWORD *)v8 + 1) = v5;
      v5 = v22[0];
      LODWORD(v25) = v25 + 1;
    }
    else
    {
      if ( v8 )
      {
        *v8 = 0;
        *((_QWORD *)v8 + 1) = v5;
        v7 = v25;
        v5 = v22[0];
      }
      LODWORD(v25) = v7 + 1;
    }
  }
  else
  {
    sub_93FB40((__int64)&v24, 0);
    v5 = v22[0];
  }
  if ( v5 )
LABEL_8:
    sub_B91220((__int64)v22, v5);
  _BitScanReverse64(&v9, 1LL << *(_BYTE *)(a1 + 3));
  v10 = sub_2A2C6D0(
          &v24,
          *(_QWORD *)(a1 - 96),
          *(_BYTE **)(a1 - 64),
          *(_QWORD *)(a1 - 32),
          63 - ((unsigned __int8)v9 ^ 0x3Fu));
  v11 = *(__int64 ***)(a1 + 8);
  v21 = 0;
  v18 = (_BYTE *)v10;
  v19 = v12;
  v23 = 257;
  v13 = sub_ACADE0(v11);
  v14 = sub_2466140((__int64 *)&v24, v13, v18, &v21, 1, (__int64)v22);
  v23 = 257;
  v21 = 1;
  v15 = sub_2466140((__int64 *)&v24, v14, v19, &v21, 1, (__int64)v22);
  sub_BD84D0(a1, v15);
  sub_B43D60((_QWORD *)a1);
  nullsub_61();
  v39 = &unk_49DA100;
  nullsub_63();
  if ( v24 != (unsigned int *)v26 )
    _libc_free((unsigned __int64)v24);
  return 1;
}
