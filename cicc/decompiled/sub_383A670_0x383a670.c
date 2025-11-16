// Function: sub_383A670
// Address: 0x383a670
//
__int64 __fastcall sub_383A670(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // r14
  __int64 v6; // rdx
  __int64 v7; // r15
  char v8; // si
  __int64 v9; // rdi
  int v10; // ecx
  unsigned int v11; // r9d
  __int64 v12; // rax
  int v13; // r8d
  unsigned __int16 *v14; // rdx
  __int64 v15; // r14
  __int64 v17; // rdx
  __int64 v18; // rax
  int v19; // eax
  int v20; // r10d
  __int64 v21; // rax
  __int128 v22; // [rsp-30h] [rbp-90h]
  __int128 v23; // [rsp-20h] [rbp-80h]
  int *v24; // [rsp+8h] [rbp-58h]
  int v25; // [rsp+1Ch] [rbp-44h] BYREF
  __int64 v26; // [rsp+20h] [rbp-40h] BYREF
  int v27; // [rsp+28h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 80);
  v26 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v26, v4, 1);
  v27 = *(_DWORD *)(a2 + 72);
  v5 = sub_37AE0F0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v7 = v6;
  v25 = sub_375D5B0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v24 = sub_3805BC0(a1 + 712, &v25);
  sub_37593F0(a1, v24);
  v8 = *(_BYTE *)(a1 + 512) & 1;
  if ( v8 )
  {
    v9 = a1 + 520;
    v10 = 7;
  }
  else
  {
    v17 = *(unsigned int *)(a1 + 528);
    v9 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v17 )
      goto LABEL_12;
    v10 = v17 - 1;
  }
  v11 = v10 & (37 * *v24);
  v12 = v9 + 24LL * v11;
  v13 = *(_DWORD *)v12;
  if ( *v24 == *(_DWORD *)v12 )
    goto LABEL_6;
  v19 = 1;
  while ( v13 != -1 )
  {
    v20 = v19 + 1;
    v21 = v10 & (v11 + v19);
    v11 = v21;
    v12 = v9 + 24 * v21;
    v13 = *(_DWORD *)v12;
    if ( *v24 == *(_DWORD *)v12 )
      goto LABEL_6;
    v19 = v20;
  }
  if ( v8 )
  {
    v18 = 192;
    goto LABEL_13;
  }
  v17 = *(unsigned int *)(a1 + 528);
LABEL_12:
  v18 = 24 * v17;
LABEL_13:
  v12 = v9 + v18;
LABEL_6:
  v14 = (unsigned __int16 *)(*(_QWORD *)(v5 + 48) + 16LL * (unsigned int)v7);
  *((_QWORD *)&v23 + 1) = *(unsigned int *)(v12 + 16);
  *(_QWORD *)&v23 = *(_QWORD *)(v12 + 8);
  *((_QWORD *)&v22 + 1) = v7;
  *(_QWORD *)&v22 = v5;
  v15 = sub_340F900(
          *(_QWORD **)(a1 + 8),
          0xA6u,
          (__int64)&v26,
          *v14,
          *((_QWORD *)v14 + 1),
          *(_QWORD *)(a1 + 8),
          v22,
          v23,
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
  if ( v26 )
    sub_B91220((__int64)&v26, v26);
  return v15;
}
