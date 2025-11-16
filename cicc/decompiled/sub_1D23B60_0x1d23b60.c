// Function: sub_1D23B60
// Address: 0x1d23b60
//
void __fastcall sub_1D23B60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v8; // rax
  unsigned int v9; // edx
  char v10; // cl
  _QWORD *v11; // rsi
  char v12; // bl
  __int64 *v13; // rdi
  __int64 v14; // rax
  __int64 (*v15)(); // r8
  __int64 (*v16)(); // rax
  _QWORD **v17; // rax
  int v18; // ecx
  __int64 v19; // rax
  _QWORD *v20; // rdx
  __int64 *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdi
  char v24; // al

  if ( !a4 )
  {
    v8 = 0;
LABEL_12:
    v10 = 0;
    if ( !*(_DWORD *)(a1 + 472) )
      goto LABEL_4;
    goto LABEL_13;
  }
  v8 = a4 - 1;
  if ( a4 == 1 )
    goto LABEL_12;
  _BitScanReverse64(&v8, v8);
  v9 = 64 - (v8 ^ 0x3F);
  v10 = 64 - (v8 ^ 0x3F);
  v8 = v9;
  if ( *(_DWORD *)(a1 + 472) <= v9 )
    goto LABEL_4;
LABEL_13:
  v17 = (_QWORD **)(*(_QWORD *)(a1 + 464) + 8 * v8);
  v11 = *v17;
  if ( *v17 )
  {
    *v17 = (_QWORD *)*v11;
    if ( !a4 )
      goto LABEL_5;
    goto LABEL_15;
  }
LABEL_4:
  v11 = (_QWORD *)sub_145CBF0((__int64 *)(a1 + 360), 40LL << v10, 8);
  if ( !a4 )
  {
LABEL_5:
    v12 = 0;
    goto LABEL_6;
  }
LABEL_15:
  v18 = 0;
  v12 = 0;
  v19 = 0;
  do
  {
    v20 = &v11[5 * v19];
    v21 = (__int64 *)(a3 + 16 * v19);
    v20[2] = a2;
    *v20 = *v21;
    *((_DWORD *)v20 + 2) = *((_DWORD *)v21 + 2);
    v22 = *v21;
    v23 = *(_QWORD *)(v22 + 48);
    v20[4] = v23;
    if ( v23 )
      *(_QWORD *)(v23 + 24) = v20 + 4;
    v20[3] = v22 + 48;
    *(_QWORD *)(v22 + 48) = v20;
    if ( *(_BYTE *)(*(_QWORD *)(*v20 + 40LL) + 16LL * *((unsigned int *)v20 + 2)) != 1 && !v12 )
      v12 = (*(_BYTE *)(*v20 + 26LL) & 4) != 0;
    v19 = (unsigned int)++v18;
  }
  while ( v18 != a4 );
LABEL_6:
  *(_DWORD *)(a2 + 56) = a4;
  *(_QWORD *)(a2 + 32) = v11;
  v13 = *(__int64 **)(a1 + 16);
  v14 = *v13;
  v15 = *(__int64 (**)())(*v13 + 984);
  if ( v15 != sub_1D12E30 )
  {
    v24 = ((__int64 (__fastcall *)(__int64 *, __int64, _QWORD, _QWORD))v15)(
            v13,
            a2,
            *(_QWORD *)(a1 + 72),
            *(_QWORD *)(a1 + 64));
    v13 = *(__int64 **)(a1 + 16);
    v12 |= v24;
    v14 = *v13;
  }
  v16 = *(__int64 (**)())(v14 + 992);
  if ( v16 == sub_1D12E40 || !((unsigned __int8 (__fastcall *)(__int64 *, __int64))v16)(v13, a2) )
    *(_BYTE *)(a2 + 26) = (4 * (v12 & 1)) | *(_BYTE *)(a2 + 26) & 0xFB;
  nullsub_686();
}
