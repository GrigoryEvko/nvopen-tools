// Function: sub_2474260
// Address: 0x2474260
//
void __fastcall sub_2474260(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // rdx
  bool v6; // cc
  unsigned __int64 v7; // rdx
  char v8; // r10
  __int64 v9; // r15
  __int64 v10; // rsi
  _QWORD *v11; // rax
  char v12; // r10
  __int64 v13; // rcx
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rsi
  _BYTE *v18; // rax
  unsigned __int64 v19; // rax
  _BYTE *v20; // r15
  _BYTE *v21; // rax
  __int64 v22; // rax
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  int v31; // [rsp+8h] [rbp-188h]
  char v32; // [rsp+17h] [rbp-179h]
  _BYTE *v33; // [rsp+18h] [rbp-178h]
  char v34; // [rsp+20h] [rbp-170h]
  __int64 **v35; // [rsp+20h] [rbp-170h]
  __int64 v36; // [rsp+20h] [rbp-170h]
  char v37; // [rsp+20h] [rbp-170h]
  __int64 v38; // [rsp+28h] [rbp-168h]
  int v39; // [rsp+38h] [rbp-158h]
  _BYTE v40[32]; // [rsp+40h] [rbp-150h] BYREF
  __int16 v41; // [rsp+60h] [rbp-130h]
  _BYTE v42[32]; // [rsp+70h] [rbp-120h] BYREF
  __int16 v43; // [rsp+90h] [rbp-100h]
  _QWORD v44[4]; // [rsp+A0h] [rbp-F0h] BYREF
  __int16 v45; // [rsp+C0h] [rbp-D0h]
  unsigned int *v46[24]; // [rsp+D0h] [rbp-C0h] BYREF

  sub_23D0AB0((__int64)v46, a2, 0, 0, 0);
  v3 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v4 = *(_QWORD *)(a2 - 32 * v3);
  v5 = *(_QWORD *)(a2 + 32 * (1 - v3));
  v6 = *(_DWORD *)(v5 + 32) <= 0x40u;
  v7 = *(_QWORD *)(v5 + 24);
  if ( !v6 )
    v7 = *(_QWORD *)v7;
  v8 = -1;
  if ( v7 )
  {
    _BitScanReverse64(&v7, v7);
    v8 = 63 - (v7 ^ 0x3F);
  }
  v9 = *(_QWORD *)(a2 + 32 * (2 - v3));
  v38 = *(_QWORD *)(a2 + 32 * (3 - v3));
  if ( (_BYTE)qword_4FE84C8 )
  {
    v37 = v8;
    sub_2472230(a1, v4, a2);
    sub_2472230(a1, v9, a2);
    v8 = v37;
  }
  v10 = *(_QWORD *)(a2 + 8);
  if ( !*(_BYTE *)(a1 + 633) )
  {
    v28 = sub_2463540((__int64 *)a1, v10);
    v29 = (__int64)v28;
    if ( v28 )
      v29 = sub_AD6530((__int64)v28, v10);
    sub_246EF60(a1, a2, v29);
    v30 = sub_AD6530(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 88LL), a2);
    v17 = a2;
    sub_246F1C0(a1, a2, v30);
    goto LABEL_9;
  }
  v34 = v8;
  v11 = sub_2463540((__int64 *)a1, v10);
  v12 = v34;
  v13 = (__int64)v11;
  BYTE1(v11) = 1;
  v35 = (__int64 **)v13;
  LOBYTE(v11) = v12;
  v32 = v12;
  v33 = sub_2466120(a1, v4, v46, v13, (unsigned __int16)v11, 0);
  v31 = v14;
  v44[0] = "_msmaskedld";
  v45 = 259;
  v15 = sub_246F3F0(a1, v38);
  v16 = sub_B34C20((__int64)v46, v35, (__int64)v33, v32, v9, v15, (__int64)v44);
  v17 = a2;
  sub_246EF60(a1, a2, v16);
  if ( !*(_DWORD *)(*(_QWORD *)(a1 + 8) + 4LL) )
  {
LABEL_9:
    sub_F94A20(v46, v17);
    return;
  }
  v43 = 257;
  v41 = 257;
  v45 = 257;
  v18 = (_BYTE *)sub_AD6530(*(_QWORD *)(v9 + 8), a2);
  v19 = sub_929DE0(v46, v18, (_BYTE *)v9, (__int64)v40, 0, 0);
  v20 = (_BYTE *)sub_24633A0((__int64 *)v46, 0x28u, v19, v35, (__int64)v42, 0, v39, 0);
  v21 = (_BYTE *)sub_246F3F0(a1, v38);
  v22 = sub_A82350(v46, v21, v20, (__int64)v44);
  v44[0] = "_mscmp";
  v45 = 259;
  v23 = sub_2465600(a1, v22, (__int64)v46, (__int64)v44);
  v24 = *(_QWORD *)(a1 + 8);
  v45 = 257;
  v25 = sub_A82CA0(v46, *(_QWORD *)(v24 + 88), v31, 0, 0, (__int64)v44);
  v45 = 257;
  v36 = v25;
  v26 = sub_246EE10(a1, v38);
  v27 = sub_B36550(v46, v23, v26, v36, (__int64)v44, 0);
  sub_246F1C0(a1, a2, v27);
  sub_F94A20(v46, a2);
}
