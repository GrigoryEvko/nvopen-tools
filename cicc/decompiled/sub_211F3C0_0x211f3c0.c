// Function: sub_211F3C0
// Address: 0x211f3c0
//
__int64 __fastcall sub_211F3C0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // rax
  unsigned __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  char v12; // cl
  __int64 v13; // rax
  bool v14; // al
  __int64 v15; // rsi
  _QWORD *v16; // r13
  __int64 v17; // rcx
  __int64 v18; // r10
  __int64 v19; // r11
  __int64 v20; // r14
  __int64 v21; // [rsp+0h] [rbp-A0h]
  __int64 v22; // [rsp+0h] [rbp-A0h]
  __int64 v23; // [rsp+8h] [rbp-98h]
  __int64 v24; // [rsp+18h] [rbp-88h]
  __int64 v25; // [rsp+20h] [rbp-80h]
  __int64 v26; // [rsp+28h] [rbp-78h]
  __int64 v27; // [rsp+30h] [rbp-70h] BYREF
  int v28; // [rsp+38h] [rbp-68h]
  __int64 v29; // [rsp+40h] [rbp-60h] BYREF
  __int64 v30; // [rsp+48h] [rbp-58h]
  __int64 v31; // [rsp+50h] [rbp-50h] BYREF
  __int64 v32; // [rsp+58h] [rbp-48h]

  if ( *(_WORD *)(a2 + 24) == 186 && (*(_BYTE *)(a2 + 27) & 4) == 0 && (*(_WORD *)(a2 + 26) & 0x380) == 0 )
    return sub_2146690();
  v4 = *(_QWORD *)(a2 + 32);
  v5 = *(_QWORD *)(v4 + 80);
  v6 = *(_QWORD *)(v4 + 88);
  v26 = *(_QWORD *)v4;
  v25 = *(_QWORD *)(v4 + 8);
  sub_1F40D10(
    (__int64)&v31,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(v4 + 40) + 40LL) + 16LL * *(unsigned int *)(v4 + 48)),
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v4 + 40) + 40LL) + 16LL * *(unsigned int *)(v4 + 48) + 8));
  v7 = *(_QWORD *)(a2 + 32);
  v28 = 0;
  LODWORD(v30) = 0;
  v8 = *(_QWORD *)(v7 + 40);
  v9 = *(_QWORD *)(v7 + 48);
  v27 = 0;
  v10 = *(unsigned int *)(v7 + 48);
  v29 = 0;
  v11 = *(_QWORD *)(v8 + 40) + 16 * v10;
  v12 = *(_BYTE *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  LOBYTE(v31) = v12;
  v32 = v13;
  if ( v12 )
  {
    if ( (unsigned __int8)(v12 - 14) <= 0x47u || (unsigned __int8)(v12 - 2) <= 5u )
      goto LABEL_7;
LABEL_14:
    sub_2016B80((__int64)a1, v8, v9, &v27, &v29);
    goto LABEL_8;
  }
  v21 = v9;
  v14 = sub_1F58CF0((__int64)&v31);
  v9 = v21;
  if ( !v14 )
    goto LABEL_14;
LABEL_7:
  sub_20174B0((__int64)a1, v8, v9, &v27, &v29);
LABEL_8:
  v15 = *(_QWORD *)(a2 + 72);
  v16 = (_QWORD *)a1[1];
  v17 = *(_QWORD *)(a2 + 104);
  v18 = *(unsigned __int8 *)(a2 + 88);
  v31 = v15;
  v19 = *(_QWORD *)(a2 + 96);
  if ( v15 )
  {
    v22 = v18;
    v23 = *(_QWORD *)(a2 + 96);
    v24 = v17;
    sub_1623A60((__int64)&v31, v15, 2);
    v18 = v22;
    v19 = v23;
    v17 = v24;
  }
  LODWORD(v32) = *(_DWORD *)(a2 + 64);
  v20 = sub_1D2C2D0(v16, v26, v25, (__int64)&v31, v29, v30, v5, v6, v18, v19, v17);
  if ( v31 )
    sub_161E7C0((__int64)&v31, v31);
  return v20;
}
