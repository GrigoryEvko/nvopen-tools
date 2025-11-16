// Function: sub_1FA0740
// Address: 0x1fa0740
//
__int64 __fastcall sub_1FA0740(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        char a6,
        __int64 a7,
        __int64 a8,
        int a9,
        int a10)
{
  __int16 v12; // ax
  char v14; // al
  __int64 v15; // r11
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // r14
  __int64 v20; // rsi
  __int64 v21; // r9
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // r11
  __int64 v25; // r14
  __int64 v26; // rdx
  __int64 v27; // rbx
  __int64 v28; // [rsp+0h] [rbp-60h]
  __int64 v30; // [rsp+8h] [rbp-58h]
  __int64 v32; // [rsp+10h] [rbp-50h]
  __int64 v34; // [rsp+20h] [rbp-40h] BYREF
  __int64 v35; // [rsp+28h] [rbp-38h]
  __int64 v36; // [rsp+78h] [rbp+18h]
  __int64 v37; // [rsp+78h] [rbp+18h]

  v12 = *(_WORD *)(a8 + 24);
  if ( a10 == 2 )
  {
    if ( v12 != 185 )
      return 0;
    v14 = (*(_BYTE *)(a8 + 27) >> 2) & 3;
    if ( v14 == 2 )
      goto LABEL_6;
    goto LABEL_5;
  }
  if ( v12 != 185 )
    return 0;
  v14 = (*(_BYTE *)(a8 + 27) >> 2) & 3;
  if ( v14 != 3 )
  {
LABEL_5:
    if ( v14 != 1 )
      return 0;
  }
LABEL_6:
  if ( (*(_WORD *)(a8 + 26) & 0x380) != 0 || !sub_1D18C00(a8, 1, a9) )
    return 0;
  v15 = a8;
  v16 = a5;
  v17 = *(unsigned __int8 *)(a8 + 88);
  v18 = *(_QWORD *)(a8 + 96);
  v19 = v17;
  if ( (a6 || (*(_BYTE *)(a8 + 26) & 8) != 0)
    && (!(_BYTE)v17
     || !(_BYTE)a4
     || (((int)*(unsigned __int16 *)(a3 + 2 * (v17 + 115LL * (unsigned __int8)a4 + 16104)) >> (4 * a10)) & 0xF) != 0) )
  {
    return 0;
  }
  v20 = *(_QWORD *)(a8 + 72);
  v21 = *(_QWORD *)(a8 + 104);
  v22 = *(_QWORD *)(a8 + 32);
  v34 = v20;
  if ( v20 )
  {
    v28 = a5;
    v30 = v22;
    v32 = v21;
    sub_1623A60((__int64)&v34, v20, 2);
    v15 = a8;
    v16 = v28;
    v22 = v30;
    v21 = v32;
  }
  LODWORD(v35) = *(_DWORD *)(v15 + 64);
  v36 = v15;
  v23 = sub_1D2B590(
          a1,
          a10,
          (__int64)&v34,
          a4,
          v16,
          v21,
          *(_OWORD *)v22,
          *(_QWORD *)(v22 + 40),
          *(_QWORD *)(v22 + 48),
          v19,
          v18);
  v24 = v36;
  v25 = v23;
  v27 = v26;
  if ( v34 )
  {
    sub_161E7C0((__int64)&v34, v34);
    v24 = v36;
  }
  v37 = v24;
  v34 = v25;
  v35 = v27;
  sub_1F994A0(a2, a7, &v34, 1, 1);
  sub_1D44C70((__int64)a1, v37, 1, v25, 1u);
  return a7;
}
