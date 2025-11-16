// Function: sub_383B380
// Address: 0x383b380
//
unsigned __int8 *__fastcall sub_383B380(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  unsigned __int16 *v7; // rax
  __int64 v8; // rsi
  unsigned __int64 v9; // rcx
  unsigned int v10; // r8d
  int *v11; // rbx
  char v12; // di
  __int64 v13; // r9
  int v14; // esi
  int v15; // edx
  unsigned int v16; // r10d
  __int64 v17; // rax
  int v18; // r11d
  _QWORD *v19; // r15
  __int64 v20; // rsi
  __int64 v21; // rbx
  __int128 v22; // rax
  unsigned __int8 *v23; // r12
  __int64 v25; // rax
  __int64 v26; // rax
  int v27; // eax
  int v28; // ebx
  __int128 v29; // [rsp-20h] [rbp-80h]
  unsigned int v30; // [rsp+0h] [rbp-60h]
  unsigned int v31; // [rsp+0h] [rbp-60h]
  __int64 v32; // [rsp+0h] [rbp-60h]
  unsigned __int64 v33; // [rsp+8h] [rbp-58h]
  int v34; // [rsp+1Ch] [rbp-44h] BYREF
  __int64 v35; // [rsp+20h] [rbp-40h] BYREF
  int v36; // [rsp+28h] [rbp-38h]

  v7 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3);
  v8 = *(_QWORD *)(a2 + 80);
  v9 = *((_QWORD *)v7 + 1);
  v10 = *v7;
  v35 = v8;
  v33 = v9;
  if ( v8 )
  {
    v30 = v10;
    sub_B96E90((__int64)&v35, v8, 1);
    v10 = v30;
  }
  v31 = v10;
  v36 = *(_DWORD *)(a2 + 72);
  v34 = sub_375D5B0(a1, a2, a3);
  v11 = sub_3805BC0(a1 + 712, &v34);
  sub_37593F0(a1, v11);
  v12 = *(_BYTE *)(a1 + 512) & 1;
  if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
  {
    v13 = a1 + 520;
    v14 = 7;
  }
  else
  {
    v25 = *(unsigned int *)(a1 + 528);
    v13 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v25 )
      goto LABEL_12;
    v14 = v25 - 1;
  }
  v15 = *v11;
  v16 = v14 & (37 * *v11);
  v17 = v13 + 24LL * v16;
  v18 = *(_DWORD *)v17;
  if ( *v11 == *(_DWORD *)v17 )
    goto LABEL_6;
  v27 = 1;
  while ( v18 != -1 )
  {
    v28 = v27 + 1;
    v16 = v14 & (v27 + v16);
    v17 = v13 + 24LL * v16;
    v18 = *(_DWORD *)v17;
    if ( v15 == *(_DWORD *)v17 )
      goto LABEL_6;
    v27 = v28;
  }
  if ( v12 )
  {
    v26 = 192;
    goto LABEL_13;
  }
  v25 = *(unsigned int *)(a1 + 528);
LABEL_12:
  v26 = 24 * v25;
LABEL_13:
  v17 = v13 + v26;
LABEL_6:
  v19 = *(_QWORD **)(a1 + 8);
  v20 = v31;
  v21 = *(unsigned int *)(v17 + 16);
  v32 = *(_QWORD *)(v17 + 8);
  *(_QWORD *)&v22 = sub_33F7D60(v19, v20, v33);
  *((_QWORD *)&v29 + 1) = v21 | a3 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v29 = v32;
  v23 = sub_3406EB0(
          v19,
          0xDEu,
          (__int64)&v35,
          *(unsigned __int16 *)(*(_QWORD *)(v32 + 48) + 16 * v21),
          *(_QWORD *)(*(_QWORD *)(v32 + 48) + 16 * v21 + 8),
          v32,
          v29,
          v22);
  if ( v35 )
    sub_B91220((__int64)&v35, v35);
  return v23;
}
