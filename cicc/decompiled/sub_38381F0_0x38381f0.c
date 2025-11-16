// Function: sub_38381F0
// Address: 0x38381f0
//
unsigned __int8 *__fastcall sub_38381F0(__int64 a1, __int64 a2, __m128i a3)
{
  int *v5; // r12
  int v6; // r9d
  char v7; // si
  __int64 v8; // r8
  int v9; // ecx
  unsigned int v10; // edi
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rsi
  unsigned __int16 *v14; // rax
  __int64 v15; // r8
  __int64 v16; // rcx
  unsigned __int8 *v17; // r12
  __int64 v19; // rax
  __int64 v20; // rax
  int v21; // eax
  int v22; // r10d
  __int64 v23; // [rsp+0h] [rbp-50h]
  __int64 v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+10h] [rbp-40h] BYREF
  int v26; // [rsp+18h] [rbp-38h]

  LODWORD(v25) = sub_375D5B0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v5 = sub_3805BC0(a1 + 712, (int *)&v25);
  sub_37593F0(a1, v5);
  v7 = *(_BYTE *)(a1 + 512) & 1;
  if ( v7 )
  {
    v8 = a1 + 520;
    v9 = 7;
  }
  else
  {
    v19 = *(unsigned int *)(a1 + 528);
    v8 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v19 )
      goto LABEL_12;
    v9 = v19 - 1;
  }
  v10 = v9 & (37 * *v5);
  v11 = v8 + 24LL * v10;
  v6 = *(_DWORD *)v11;
  if ( *v5 == *(_DWORD *)v11 )
    goto LABEL_4;
  v21 = 1;
  while ( v6 != -1 )
  {
    v22 = v21 + 1;
    v10 = v9 & (v21 + v10);
    v11 = v8 + 24LL * v10;
    v6 = *(_DWORD *)v11;
    if ( *v5 == *(_DWORD *)v11 )
      goto LABEL_4;
    v21 = v22;
  }
  if ( v7 )
  {
    v20 = 192;
    goto LABEL_13;
  }
  v19 = *(unsigned int *)(a1 + 528);
LABEL_12:
  v20 = 24 * v19;
LABEL_13:
  v11 = v8 + v20;
LABEL_4:
  v12 = *(_QWORD *)(a1 + 8);
  v13 = *(_QWORD *)(a2 + 80);
  v14 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(v11 + 8) + 48LL) + 16LL * *(unsigned int *)(v11 + 16));
  v15 = *((_QWORD *)v14 + 1);
  v16 = *v14;
  v25 = v13;
  if ( v13 )
  {
    v23 = v16;
    v24 = v15;
    sub_B96E90((__int64)&v25, v13, 1);
    v16 = v23;
    v15 = v24;
  }
  v26 = *(_DWORD *)(a2 + 72);
  v17 = sub_33FAF80(v12, 52, (__int64)&v25, v16, v15, v6, a3);
  if ( v25 )
    sub_B91220((__int64)&v25, v25);
  return v17;
}
