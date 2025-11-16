// Function: sub_3836D60
// Address: 0x3836d60
//
__int64 *__fastcall sub_3836D60(__int64 a1, __int64 a2)
{
  int *v4; // r13
  char v5; // si
  __int64 v6; // r8
  int v7; // ecx
  unsigned int v8; // edi
  __int64 v9; // rax
  int v10; // r9d
  __int64 v11; // rdx
  __int64 v12; // rcx
  _QWORD *v13; // rdi
  __int64 v15; // rax
  __int64 v16; // rax
  int v17; // eax
  int v18; // r10d
  int v19[9]; // [rsp+Ch] [rbp-24h] BYREF

  v19[0] = sub_375D5B0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v4 = sub_3805BC0(a1 + 712, v19);
  sub_37593F0(a1, v4);
  v5 = *(_BYTE *)(a1 + 512) & 1;
  if ( v5 )
  {
    v6 = a1 + 520;
    v7 = 7;
  }
  else
  {
    v15 = *(unsigned int *)(a1 + 528);
    v6 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v15 )
      goto LABEL_10;
    v7 = v15 - 1;
  }
  v8 = v7 & (37 * *v4);
  v9 = v6 + 24LL * v8;
  v10 = *(_DWORD *)v9;
  if ( *v4 == *(_DWORD *)v9 )
    goto LABEL_4;
  v17 = 1;
  while ( v10 != -1 )
  {
    v18 = v17 + 1;
    v8 = v7 & (v17 + v8);
    v9 = v6 + 24LL * v8;
    v10 = *(_DWORD *)v9;
    if ( *v4 == *(_DWORD *)v9 )
      goto LABEL_4;
    v17 = v18;
  }
  if ( v5 )
  {
    v16 = 192;
    goto LABEL_11;
  }
  v15 = *(unsigned int *)(a1 + 528);
LABEL_10:
  v16 = 24 * v15;
LABEL_11:
  v9 = v6 + v16;
LABEL_4:
  v11 = *(_QWORD *)(v9 + 8);
  v12 = *(unsigned int *)(v9 + 16);
  v13 = *(_QWORD **)(a1 + 8);
  if ( *(_DWORD *)(a2 + 24) == 492 )
    return sub_33EC3B0(
             v13,
             (__int64 *)a2,
             v11,
             v12,
             *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
             *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
             *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
  else
    return sub_33EBEE0(v13, (__int64 *)a2, v11, v12);
}
