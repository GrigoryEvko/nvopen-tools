// Function: sub_3842650
// Address: 0x3842650
//
unsigned __int8 *__fastcall sub_3842650(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rsi
  int *v6; // r14
  char v7; // si
  __int64 v8; // r8
  int v9; // ecx
  unsigned int v10; // edi
  int *v11; // rax
  int v12; // r9d
  __int64 v13; // r15
  unsigned int v14; // edx
  unsigned __int8 *v15; // rsi
  unsigned __int16 *v16; // rax
  unsigned __int8 *v17; // r14
  __int64 v19; // rax
  __int64 v20; // rax
  int v21; // eax
  int v22; // r10d
  int v23; // [rsp+1Ch] [rbp-44h] BYREF
  __int64 v24; // [rsp+20h] [rbp-40h] BYREF
  int v25; // [rsp+28h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 80);
  v24 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v24, v5, 1);
  v25 = *(_DWORD *)(a2 + 72);
  v23 = sub_375D5B0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = sub_3805BC0(a1 + 712, &v23);
  sub_37593F0(a1, v6);
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
  v10 = v9 & (37 * *v6);
  v11 = (int *)(v8 + 24LL * v10);
  v12 = *v11;
  if ( *v6 == *v11 )
    goto LABEL_6;
  v21 = 1;
  while ( v12 != -1 )
  {
    v22 = v21 + 1;
    v10 = v9 & (v21 + v10);
    v11 = (int *)(v8 + 24LL * v10);
    v12 = *v11;
    if ( *v6 == *v11 )
      goto LABEL_6;
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
  v11 = (int *)(v8 + v20);
LABEL_6:
  v13 = (unsigned int)v11[4];
  v15 = sub_33FAF80(
          *(_QWORD *)(a1 + 8),
          215,
          (__int64)&v24,
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          0,
          a3);
  v16 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL)
                           + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL));
  v17 = sub_34070B0(
          *(_QWORD **)(a1 + 8),
          (__int64)v15,
          v14 | v13 & 0xFFFFFFFF00000000LL,
          (__int64)&v24,
          *v16,
          *((_QWORD *)v16 + 1),
          a3);
  if ( v24 )
    sub_B91220((__int64)&v24, v24);
  return v17;
}
