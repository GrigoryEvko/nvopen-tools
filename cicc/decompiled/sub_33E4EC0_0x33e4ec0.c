// Function: sub_33E4EC0
// Address: 0x33e4ec0
//
void __fastcall sub_33E4EC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v8; // rax
  unsigned int v9; // ecx
  __int64 v10; // rax
  _QWORD *v11; // rsi
  char *v12; // rdx
  char v13; // r14
  __int64 *v14; // rdi
  __int64 v15; // rax
  __int64 (*v16)(); // rdx
  __int64 (*v17)(); // rax
  _QWORD **v18; // rax
  int v19; // ecx
  __int64 v20; // rax
  _QWORD *v21; // rdx
  __int64 *v22; // rax
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // r8
  __int16 v26; // ax

  if ( a4 )
  {
    v8 = a4 - 1;
    if ( a4 != 1 )
    {
      _BitScanReverse64(&v8, v8);
      v9 = 64 - (v8 ^ 0x3F);
      v8 = (int)v9;
      if ( *(_DWORD *)(a1 + 648) <= v9 )
        goto LABEL_4;
      goto LABEL_15;
    }
  }
  else
  {
    v8 = 0;
  }
  LOBYTE(v9) = 0;
  if ( !*(_DWORD *)(a1 + 648) )
    goto LABEL_4;
LABEL_15:
  v18 = (_QWORD **)(*(_QWORD *)(a1 + 640) + 8 * v8);
  v11 = *v18;
  if ( *v18 )
  {
    *v18 = (_QWORD *)*v11;
    goto LABEL_17;
  }
LABEL_4:
  v10 = *(_QWORD *)(a1 + 544);
  *(_QWORD *)(a1 + 624) += 40LL << v9;
  v11 = (_QWORD *)((v10 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  v12 = (char *)v11 + (40LL << v9);
  if ( *(_QWORD *)(a1 + 552) < (unsigned __int64)v12 || !v10 )
  {
    v11 = (_QWORD *)sub_9D1E70(a1 + 544, 40LL << v9, 40LL << v9, 3);
LABEL_17:
    if ( !a4 )
      goto LABEL_7;
    goto LABEL_18;
  }
  *(_QWORD *)(a1 + 544) = v12;
  if ( !a4 )
  {
LABEL_7:
    v13 = 0;
    goto LABEL_8;
  }
LABEL_18:
  v19 = 0;
  v13 = 0;
  v20 = 0;
  do
  {
    v21 = &v11[5 * v20];
    v22 = (__int64 *)(a3 + 16 * v20);
    v21[2] = a2;
    *v21 = *v22;
    *((_DWORD *)v21 + 2) = *((_DWORD *)v22 + 2);
    v23 = *v22;
    v24 = *(_QWORD *)(v23 + 56);
    v21[4] = v24;
    if ( v24 )
      *(_QWORD *)(v24 + 24) = v21 + 4;
    v21[3] = v23 + 56;
    *(_QWORD *)(v23 + 56) = v21;
    v25 = *v21;
    v26 = *(_WORD *)(*(_QWORD *)(*v21 + 48LL) + 16LL * *((unsigned int *)v21 + 2));
    if ( v26 != 1 && (v26 != 262 || (unsigned int)(*(_DWORD *)(v25 + 24) - 49) > 1) && (*(_BYTE *)(v25 + 32) & 4) != 0 )
      v13 = 1;
    v20 = (unsigned int)++v19;
  }
  while ( v19 != a4 );
LABEL_8:
  *(_DWORD *)(a2 + 64) = a4;
  *(_QWORD *)(a2 + 40) = v11;
  v14 = *(__int64 **)(a1 + 16);
  v15 = *v14;
  v16 = *(__int64 (**)())(*v14 + 1880);
  if ( v16 != sub_302E050 )
  {
    if ( ((unsigned __int8 (__fastcall *)(__int64 *, __int64))v16)(v14, a2) )
      goto LABEL_12;
    v14 = *(__int64 **)(a1 + 16);
    v15 = *v14;
  }
  v17 = *(__int64 (**)())(v15 + 1856);
  if ( v17 != sub_302E040 )
    v13 |= ((__int64 (__fastcall *)(__int64 *, __int64, _QWORD, _QWORD))v17)(
             v14,
             a2,
             *(_QWORD *)(a1 + 88),
             *(_QWORD *)(a1 + 80));
  *(_BYTE *)(a2 + 32) = *(_BYTE *)(a2 + 32) & 0xFB | (4 * (v13 & 1));
LABEL_12:
  nullsub_1875();
}
