// Function: sub_3271140
// Address: 0x3271140
//
__int64 __fastcall sub_3271140(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v8; // rdx
  __int64 v10; // r8
  int v11; // r10d
  char v13; // bl
  int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // r9
  __int128 *v18; // rcx
  __int64 v19; // rdi
  int v20; // r11d
  __int64 v21; // rax
  int v22; // r9d
  int v23; // r11d
  __int64 v24; // r14
  __int64 v25; // rcx
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rdx
  int v29; // [rsp+0h] [rbp-80h]
  int v30; // [rsp+8h] [rbp-78h]
  __int128 *v31; // [rsp+10h] [rbp-70h]
  int v32; // [rsp+18h] [rbp-68h]
  int v33; // [rsp+18h] [rbp-68h]
  int v34; // [rsp+20h] [rbp-60h]
  int v35; // [rsp+20h] [rbp-60h]
  __int64 v36; // [rsp+40h] [rbp-40h] BYREF
  int v37; // [rsp+48h] [rbp-38h]

  if ( *(_DWORD *)(a5 + 24) != 338 )
    return 0;
  v8 = *(unsigned __int16 *)(a5 + 96);
  v10 = *(_QWORD *)(a5 + 104);
  v11 = (unsigned __int16)v8;
  if ( !(_WORD)a3 )
    return 0;
  if ( !(_WORD)v8 )
    return 0;
  v13 = a6;
  if ( (((int)*(unsigned __int16 *)(a2 + 2 * (v8 + 274LL * (unsigned __int16)a3 + 146776) + 14) >> (4 * a6)) & 0xF) != 0 )
    return 0;
  v15 = (*(_BYTE *)(a5 + 33) >> 2) & 3;
  if ( v15 == 3 && a6 == 2 )
    return 0;
  if ( v15 == 2 && a6 == 3 )
    return 0;
  v16 = *(_QWORD *)(a5 + 80);
  v17 = *(_QWORD *)(a5 + 112);
  v18 = *(__int128 **)(a5 + 40);
  v19 = *(_QWORD *)(*(_QWORD *)(a5 + 48) + 8LL);
  v20 = **(unsigned __int16 **)(a5 + 48);
  v36 = v16;
  if ( v16 )
  {
    v29 = v20;
    v30 = (unsigned __int16)v8;
    v31 = v18;
    v32 = v17;
    v34 = v10;
    sub_B96E90((__int64)&v36, v16, 1);
    v20 = v29;
    v11 = v30;
    v18 = v31;
    LODWORD(v17) = v32;
    LODWORD(v10) = v34;
  }
  v35 = v20;
  v37 = *(_DWORD *)(a5 + 72);
  v21 = sub_33E6F50(a1, 338, (unsigned int)&v36, v11, v10, v17, a3, a4, *v18, *(__int128 *)((char *)v18 + 40));
  v23 = v35;
  v24 = v21;
  if ( v36 )
  {
    sub_B91220((__int64)&v36, v36);
    v23 = v35;
  }
  v25 = v24;
  *(_BYTE *)(v24 + 33) = *(_BYTE *)(v24 + 33) & 0xF3 | (4 * (v13 & 3));
  v26 = *(_QWORD *)(a5 + 80);
  v36 = v26;
  if ( v26 )
  {
    v33 = v23;
    sub_B96E90((__int64)&v36, v26, 1);
    v23 = v33;
    v25 = v24;
  }
  v37 = *(_DWORD *)(a5 + 72);
  v27 = sub_33FAF80(a1, 216, (unsigned int)&v36, v23, v19, v22, (unsigned __int64)v25);
  sub_34161C0(a1, a5, 0, v27, v28);
  if ( v36 )
    sub_B91220((__int64)&v36, v36);
  sub_34161C0(a1, a5, 1, v24, 1);
  return v24;
}
