// Function: sub_38383A0
// Address: 0x38383a0
//
unsigned __int8 *__fastcall sub_38383A0(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rsi
  int *v6; // r13
  char v7; // si
  __int64 v8; // r8
  int v9; // ecx
  unsigned int v10; // edi
  __int64 v11; // rax
  int v12; // r9d
  unsigned __int8 *v13; // r14
  __int64 v15; // rax
  __int64 v16; // rax
  int v17; // eax
  int v18; // r10d
  int v19; // [rsp+Ch] [rbp-44h] BYREF
  __int64 v20; // [rsp+10h] [rbp-40h] BYREF
  int v21; // [rsp+18h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 80);
  v20 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v20, v5, 1);
  v21 = *(_DWORD *)(a2 + 72);
  v19 = sub_375D5B0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = sub_3805BC0(a1 + 712, &v19);
  sub_37593F0(a1, v6);
  v7 = *(_BYTE *)(a1 + 512) & 1;
  if ( v7 )
  {
    v8 = a1 + 520;
    v9 = 7;
  }
  else
  {
    v15 = *(unsigned int *)(a1 + 528);
    v8 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v15 )
      goto LABEL_12;
    v9 = v15 - 1;
  }
  v10 = v9 & (37 * *v6);
  v11 = v8 + 24LL * v10;
  v12 = *(_DWORD *)v11;
  if ( *v6 == *(_DWORD *)v11 )
    goto LABEL_6;
  v17 = 1;
  while ( v12 != -1 )
  {
    v18 = v17 + 1;
    v10 = v9 & (v17 + v10);
    v11 = v8 + 24LL * v10;
    v12 = *(_DWORD *)v11;
    if ( *v6 == *(_DWORD *)v11 )
      goto LABEL_6;
    v17 = v18;
  }
  if ( v7 )
  {
    v16 = 192;
    goto LABEL_13;
  }
  v15 = *(unsigned int *)(a1 + 528);
LABEL_12:
  v16 = 24 * v15;
LABEL_13:
  v11 = v8 + v16;
LABEL_6:
  v13 = sub_33FAF80(
          *(_QWORD *)(a1 + 8),
          164,
          (__int64)&v20,
          *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(v11 + 8) + 48LL) + 16LL * *(unsigned int *)(v11 + 16)),
          *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v11 + 8) + 48LL) + 16LL * *(unsigned int *)(v11 + 16) + 8),
          *(_QWORD *)(a1 + 8),
          a3);
  if ( v20 )
    sub_B91220((__int64)&v20, v20);
  return v13;
}
