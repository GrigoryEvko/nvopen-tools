// Function: sub_21DF050
// Address: 0x21df050
//
__int64 __fastcall sub_21DF050(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r14
  __int64 v4; // r15
  __int64 v5; // rax
  unsigned int v6; // edx
  __int64 *v7; // r13
  __int64 v8; // r13
  __int64 v9; // rax
  unsigned int v10; // edx
  unsigned __int8 v11; // al
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r9
  __int64 v15; // rsi
  __int64 v16; // r10
  __int64 v17; // r11
  __int16 v18; // r12
  __int64 v19; // r8
  unsigned int v20; // r15d
  __int64 v21; // r12
  __int128 v23; // [rsp-10h] [rbp-70h]
  __int64 v24; // [rsp+0h] [rbp-60h]
  __int64 v25; // [rsp+8h] [rbp-58h]
  __int64 v26; // [rsp+18h] [rbp-48h]
  __int64 v27; // [rsp+20h] [rbp-40h] BYREF
  int v28; // [rsp+28h] [rbp-38h]

  v3 = *(_QWORD **)(a1 - 176);
  v4 = v3[2];
  v5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 88LL);
  v6 = *(_DWORD *)(v5 + 32);
  v7 = *(__int64 **)(v5 + 24);
  if ( v6 > 0x40 )
    v8 = *v7;
  else
    v8 = (__int64)((_QWORD)v7 << (64 - (unsigned __int8)v6)) >> (64 - (unsigned __int8)v6);
  v9 = sub_1E0A0C0(v3[4]);
  v10 = 8 * sub_15A9520(v9, 0);
  if ( v10 == 32 )
  {
    v11 = 5;
  }
  else if ( v10 > 0x20 )
  {
    v11 = 6;
    if ( v10 != 64 )
    {
      v11 = 0;
      if ( v10 == 128 )
        v11 = 7;
    }
  }
  else
  {
    v11 = 3;
    if ( v10 != 8 )
      v11 = 4 * (v10 == 16);
  }
  v12 = sub_21D74F0(v4, v3, v8, v11, 0);
  v15 = *(_QWORD *)(a2 + 72);
  v16 = v12;
  v17 = v13;
  v18 = 3183 - ((*(_BYTE *)(*(_QWORD *)(a1 + 16) + 936LL) == 0) - 1);
  v19 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL);
  v20 = **(unsigned __int8 **)(a2 + 40);
  v27 = v15;
  if ( v15 )
  {
    v25 = v13;
    v24 = v12;
    v26 = v19;
    sub_1623A60((__int64)&v27, v15, 2);
    v16 = v24;
    v17 = v25;
    v19 = v26;
  }
  *((_QWORD *)&v23 + 1) = v17;
  *(_QWORD *)&v23 = v16;
  v28 = *(_DWORD *)(a2 + 64);
  v21 = sub_1D2CC80(v3, v18, (__int64)&v27, v20, v19, v14, v23);
  if ( v27 )
    sub_161E7C0((__int64)&v27, v27);
  return v21;
}
