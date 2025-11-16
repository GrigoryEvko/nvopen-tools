// Function: sub_28CF050
// Address: 0x28cf050
//
_QWORD *__fastcall sub_28CF050(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rax
  _QWORD *v6; // r12
  unsigned __int64 v7; // rax
  unsigned int v8; // ecx
  _QWORD **v9; // rdx
  _QWORD *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v17; // rdx
  __int64 v18; // rsi
  char *v19; // rcx
  __int64 v20; // [rsp+8h] [rbp-38h]

  v20 = sub_28C86C0(a1, *(_QWORD *)(a2 - 64));
  v5 = (_QWORD *)sub_A777F0(0x48u, (__int64 *)(a1 + 72));
  v6 = v5;
  if ( v5 )
  {
    v5[2] = 0;
    v5[1] = 0xFFFFFFFD0000000CLL;
    v5[3] = 0;
    v5[4] = 2;
    v5[5] = 0;
    v5[6] = a3;
    v5[7] = a2;
    *v5 = &unk_4A21AE8;
    v5[8] = v20;
    v7 = 1;
    goto LABEL_3;
  }
  v7 = MEMORY[0x20];
  if ( MEMORY[0x20] )
  {
    v7 = MEMORY[0x20] - 1LL;
    if ( MEMORY[0x20] != 1 )
    {
LABEL_3:
      _BitScanReverse64(&v7, v7);
      v8 = 64 - (v7 ^ 0x3F);
      v7 = (int)v8;
      if ( *(_DWORD *)(a1 + 176) <= v8 )
        goto LABEL_9;
      goto LABEL_4;
    }
  }
  LOBYTE(v8) = 0;
  if ( !*(_DWORD *)(a1 + 176) )
    goto LABEL_9;
LABEL_4:
  v9 = (_QWORD **)(*(_QWORD *)(a1 + 168) + 8 * v7);
  v10 = *v9;
  if ( *v9 )
  {
    *v9 = (_QWORD *)*v10;
    goto LABEL_6;
  }
LABEL_9:
  v17 = *(_QWORD *)(a1 + 72);
  v18 = 8LL << v8;
  *(_QWORD *)(a1 + 152) += 8LL << v8;
  v10 = (_QWORD *)((v17 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  v19 = (char *)v10 + (8LL << v8);
  if ( *(_QWORD *)(a1 + 80) >= (unsigned __int64)v19 && v17 )
    *(_QWORD *)(a1 + 72) = v19;
  else
    v10 = (_QWORD *)sub_9D1E70(a1 + 72, v18, v18, 3);
LABEL_6:
  v6[3] = v10;
  v11 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
  *((_DWORD *)v6 + 3) = 0;
  v6[5] = v11;
  v12 = sub_28C86C0(a1, *(_QWORD *)(a2 - 32));
  v13 = v6[3];
  v14 = v12;
  v15 = *((unsigned int *)v6 + 9);
  *((_DWORD *)v6 + 9) = v15 + 1;
  *(_QWORD *)(v13 + 8 * v15) = v14;
  return v6;
}
