// Function: sub_32682B0
// Address: 0x32682b0
//
__int64 __fastcall sub_32682B0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rcx
  unsigned __int16 *v7; // rdx
  int v8; // eax
  __int64 v9; // rdx
  unsigned __int16 v10; // ax
  __int64 v11; // rdx
  __int64 v12; // rcx
  int v13; // edx
  __int64 result; // rax
  __int64 v15; // rdx
  _QWORD *v16; // r14
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r13
  __int64 v20; // r11
  __int64 v21; // r14
  char v22; // dl
  __int64 v23; // r15
  _BOOL4 v24; // esi
  _QWORD *v25; // rdx
  __int64 v26; // rsi
  __int128 *v27; // r10
  __int128 *v28; // r9
  __int128 *v29; // rcx
  _QWORD *v30; // r8
  __int64 v31; // [rsp+0h] [rbp-A0h]
  __int128 *v32; // [rsp+8h] [rbp-98h]
  __int128 *v33; // [rsp+10h] [rbp-90h]
  __int128 *v34; // [rsp+18h] [rbp-88h]
  _QWORD *v35; // [rsp+20h] [rbp-80h]
  _BOOL4 v36; // [rsp+28h] [rbp-78h]
  _BOOL4 v37; // [rsp+2Ch] [rbp-74h]
  int v38; // [rsp+30h] [rbp-70h]
  __int64 *v39; // [rsp+38h] [rbp-68h]
  __int64 v40; // [rsp+38h] [rbp-68h]
  unsigned __int16 v41; // [rsp+40h] [rbp-60h] BYREF
  __int64 v42; // [rsp+48h] [rbp-58h]
  unsigned __int64 v43; // [rsp+50h] [rbp-50h] BYREF
  int v44; // [rsp+58h] [rbp-48h]
  __int64 v45; // [rsp+60h] [rbp-40h] BYREF
  __int64 v46; // [rsp+68h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 40);
  v7 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(v6 + 40) + 48LL) + 16LL * *(unsigned int *)(v6 + 48));
  v8 = *v7;
  v9 = *((_QWORD *)v7 + 1);
  LOWORD(v45) = v8;
  v46 = v9;
  if ( (_WORD)v8 )
  {
    v10 = word_4456580[v8 - 1];
    v11 = 0;
  }
  else
  {
    v10 = sub_3009970((__int64)&v45, a2, v9, v6, a5);
    v6 = *(_QWORD *)(a2 + 40);
  }
  v12 = *(_QWORD *)(v6 + 160);
  v42 = v11;
  v41 = v10;
  v13 = *(_DWORD *)(v12 + 24);
  if ( v13 != 35 && v13 != 11 )
    return 0;
  v15 = *(_QWORD *)(v12 + 96);
  v16 = *(_QWORD **)(v15 + 24);
  if ( *(_DWORD *)(v15 + 32) > 0x40u )
    v16 = (_QWORD *)*v16;
  if ( v10 )
  {
    if ( v10 == 1 || (unsigned __int16)(v10 - 504) <= 7u )
      BUG();
    v18 = 16LL * (v10 - 1) + 71615648;
    v17 = *(_QWORD *)&byte_444C4A0[16 * v10 - 16];
    LOBYTE(v18) = *(_BYTE *)(v18 + 8);
  }
  else
  {
    v17 = sub_3007260((__int64)&v41);
    v45 = v17;
    v46 = v18;
  }
  LOBYTE(v44) = v18;
  v43 = (unsigned __int64)(v17 + 7) >> 3;
  if ( (_QWORD *)sub_CA1930(&v43) != v16 )
    return 0;
  v19 = *a1;
  v20 = *(_QWORD *)(a2 + 112);
  v21 = *(unsigned __int16 *)(a2 + 96);
  v22 = *(_BYTE *)(a2 + 33);
  v23 = *(_QWORD *)(a2 + 104);
  v36 = (v22 & 4) != 0;
  v24 = (v22 & 8) != 0;
  v25 = *(_QWORD **)(a2 + 40);
  v37 = v24;
  v26 = *(_QWORD *)(a2 + 80);
  v27 = (__int128 *)(v25 + 30);
  v28 = (__int128 *)(v25 + 25);
  v38 = (*(_WORD *)(a2 + 32) >> 7) & 7;
  v29 = (__int128 *)(v25 + 15);
  v39 = v25 + 10;
  v30 = v25 + 5;
  v43 = v26;
  if ( v26 )
  {
    v31 = v20;
    v32 = (__int128 *)(v25 + 30);
    v33 = (__int128 *)(v25 + 25);
    v34 = (__int128 *)(v25 + 15);
    v35 = v25 + 5;
    sub_B96E90((__int64)&v43, v26, 1);
    v25 = *(_QWORD **)(a2 + 40);
    v30 = v35;
    v20 = v31;
    v27 = v32;
    v28 = v33;
    v29 = v34;
  }
  v44 = *(_DWORD *)(a2 + 72);
  result = sub_33F51B0(
             v19,
             *v25,
             v25[1],
             (unsigned int)&v43,
             *v30,
             v30[1],
             *v39,
             v39[1],
             *v29,
             *v28,
             *v27,
             v21,
             v23,
             v20,
             v38,
             v36,
             v37);
  if ( v43 )
  {
    v40 = result;
    sub_B91220((__int64)&v43, v43);
    return v40;
  }
  return result;
}
