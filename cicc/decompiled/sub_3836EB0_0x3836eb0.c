// Function: sub_3836EB0
// Address: 0x3836eb0
//
unsigned __int8 *__fastcall sub_3836EB0(__int64 a1, __int64 a2, __m128i a3)
{
  int *v5; // r14
  __int64 v7; // r8
  int v8; // ecx
  unsigned int v9; // edi
  int *v10; // rax
  int v11; // r9d
  __int64 v12; // rsi
  __int64 v13; // r9
  __int64 v14; // r8
  unsigned int v15; // ebx
  unsigned __int8 *v16; // r14
  int v18; // eax
  int i; // eax
  int v20; // r10d
  int *v21; // rax
  __int64 v22; // [rsp+0h] [rbp-50h]
  __int64 v23; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+10h] [rbp-40h] BYREF
  int v25; // [rsp+18h] [rbp-38h]

  LODWORD(v24) = sub_375D5B0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v5 = sub_3805BC0(a1 + 712, (int *)&v24);
  sub_37593F0(a1, v5);
  if ( *(_BYTE *)(a1 + 512) & 1 )
  {
    v7 = a1 + 520;
    v8 = 7;
  }
  else
  {
    v18 = *(_DWORD *)(a1 + 528);
    v7 = *(_QWORD *)(a1 + 520);
    if ( !v18 )
      goto LABEL_4;
    v8 = v18 - 1;
  }
  v9 = v8 & (37 * *v5);
  v10 = (int *)(v7 + 24LL * v9);
  v11 = *v10;
  if ( *v5 != *v10 )
  {
    for ( i = 1; v11 != -1; i = v20 )
    {
      v20 = i + 1;
      v9 = v8 & (i + v9);
      v21 = (int *)(v7 + 24LL * v9);
      v11 = *v21;
      if ( *v5 == *v21 )
        break;
    }
  }
LABEL_4:
  v12 = *(_QWORD *)(a2 + 80);
  v13 = *(_QWORD *)(a1 + 8);
  v14 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v15 = **(unsigned __int16 **)(a2 + 48);
  v24 = v12;
  if ( v12 )
  {
    v22 = v14;
    v23 = v13;
    sub_B96E90((__int64)&v24, v12, 1);
    v14 = v22;
    v13 = v23;
  }
  v25 = *(_DWORD *)(a2 + 72);
  v16 = sub_33FAF80(v13, 215, (__int64)&v24, v15, v14, v13, a3);
  if ( v24 )
    sub_B91220((__int64)&v24, v24);
  return v16;
}
