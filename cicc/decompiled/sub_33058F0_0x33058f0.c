// Function: sub_33058F0
// Address: 0x33058f0
//
__int64 __fastcall sub_33058F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int16 *v6; // rdx
  int v7; // eax
  __int64 v8; // rdx
  unsigned __int16 v9; // ax
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rcx
  int v13; // edx
  __int64 v15; // rdx
  _QWORD *v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int8 v19; // dl
  __int64 v20; // rsi
  __int64 v21; // r15
  __int64 v22; // rcx
  char v23; // r9
  __int64 v24; // r10
  __int64 v25; // r11
  __int64 v26; // rbx
  _BOOL4 v27; // r9d
  unsigned __int16 *v28; // rsi
  __int64 v29; // rbx
  int v30; // edx
  int v31; // r15d
  __int64 v32; // [rsp+0h] [rbp-90h]
  __int64 v33; // [rsp+8h] [rbp-88h]
  _BOOL4 v34; // [rsp+14h] [rbp-7Ch]
  __int64 v35; // [rsp+18h] [rbp-78h]
  unsigned __int16 v36; // [rsp+20h] [rbp-70h] BYREF
  __int64 v37; // [rsp+28h] [rbp-68h]
  __int64 v38; // [rsp+30h] [rbp-60h]
  __int64 v39; // [rsp+38h] [rbp-58h]
  unsigned __int64 v40; // [rsp+40h] [rbp-50h] BYREF
  __int64 v41; // [rsp+48h] [rbp-48h]
  __int64 v42; // [rsp+50h] [rbp-40h]
  int v43; // [rsp+58h] [rbp-38h]

  v6 = *(unsigned __int16 **)(a2 + 48);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  LOWORD(v40) = v7;
  v41 = v8;
  if ( (_WORD)v7 )
  {
    v9 = word_4456580[v7 - 1];
    v10 = 0;
  }
  else
  {
    v9 = sub_3009970((__int64)&v40, a2, v8, a4, a5);
  }
  v37 = v10;
  v11 = *(_QWORD *)(a2 + 40);
  v36 = v9;
  v12 = *(_QWORD *)(v11 + 120);
  v13 = *(_DWORD *)(v12 + 24);
  if ( v13 != 35 && v13 != 11 )
    return 0;
  v15 = *(_QWORD *)(v12 + 96);
  v16 = *(_QWORD **)(v15 + 24);
  if ( *(_DWORD *)(v15 + 32) > 0x40u )
    v16 = (_QWORD *)*v16;
  if ( v9 )
  {
    if ( v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
      BUG();
    v18 = 16LL * (v9 - 1) + 71615648;
    v17 = *(_QWORD *)&byte_444C4A0[16 * v9 - 16];
    LOBYTE(v18) = *(_BYTE *)(v18 + 8);
  }
  else
  {
    v17 = sub_3007260((__int64)&v36);
    v38 = v17;
    v39 = v18;
  }
  LOBYTE(v41) = v18;
  v40 = (unsigned __int64)(v17 + 7) >> 3;
  if ( (_QWORD *)sub_CA1930(&v40) != v16 )
    return 0;
  v19 = *(_BYTE *)(a2 + 33);
  v20 = *(_QWORD *)(a2 + 80);
  v21 = *a1;
  v22 = *(_QWORD *)(a2 + 112);
  v23 = *(_BYTE *)(a2 + 33);
  v24 = *(unsigned __int16 *)(a2 + 96);
  v25 = *(_QWORD *)(a2 + 104);
  v40 = v20;
  v26 = *(_QWORD *)(a2 + 40);
  v27 = (v23 & 0x10) != 0;
  if ( v20 )
  {
    v34 = v27;
    v32 = v24;
    v33 = v25;
    v35 = v22;
    sub_B96E90((__int64)&v40, v20, 1);
    v19 = *(_BYTE *)(a2 + 33);
    v27 = v34;
    v24 = v32;
    v25 = v33;
    v22 = v35;
  }
  v28 = *(unsigned __int16 **)(a2 + 48);
  LODWORD(v41) = *(_DWORD *)(a2 + 72);
  v29 = sub_33E9660(
          v21,
          (*(_WORD *)(a2 + 32) >> 7) & 7,
          (v19 >> 2) & 3,
          *v28,
          *((_QWORD *)v28 + 1),
          (unsigned int)&v40,
          *(_OWORD *)v26,
          *(_QWORD *)(v26 + 40),
          *(_QWORD *)(v26 + 48),
          *(_OWORD *)(v26 + 80),
          *(_OWORD *)(v26 + 160),
          *(_OWORD *)(v26 + 200),
          v24,
          v25,
          v22,
          v27);
  v31 = v30;
  if ( v40 )
    sub_B91220((__int64)&v40, v40);
  v40 = v29;
  LODWORD(v41) = v31;
  v42 = v29;
  v43 = 1;
  return sub_32EB790((__int64)a1, a2, (__int64 *)&v40, 2, 1);
}
