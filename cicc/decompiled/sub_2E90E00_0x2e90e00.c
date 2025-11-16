// Function: sub_2E90E00
// Address: 0x2e90e00
//
__int64 __fastcall sub_2E90E00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rsi
  unsigned __int64 v9; // rax
  unsigned int v10; // ecx
  __int64 v11; // rdx
  __int64 v12; // rsi
  _QWORD *v13; // rax
  char *v14; // rcx
  _QWORD **v15; // rdx
  const __m128i *v16; // rbx
  const __m128i *v17; // r14
  const __m128i *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 result; // rax

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  v6 = *(_QWORD *)(a3 + 16);
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 16) = v6;
  v7 = *(_QWORD *)(a3 + 48);
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 48) = v7;
  v8 = *(_QWORD *)(a3 + 56);
  *(_QWORD *)(a1 + 56) = v8;
  if ( v8 )
    sub_B96E90(a1 + 56, v8, 1);
  *(_DWORD *)(a1 + 64) = 0;
  *(_WORD *)(a1 + 68) = *(_WORD *)(a3 + 68);
  v9 = *(_DWORD *)(a3 + 40) & 0xFFFFFF;
  if ( (*(_DWORD *)(a3 + 40) & 0xFFFFFF) != 0 && (--v9, v9) )
  {
    _BitScanReverse64(&v9, v9);
    v10 = 64 - (v9 ^ 0x3F);
    v9 = (int)v10;
    *(_BYTE *)(a1 + 43) = v10;
    if ( *(_DWORD *)(a2 + 240) <= v10 )
      goto LABEL_6;
  }
  else
  {
    LOBYTE(v10) = 0;
    *(_BYTE *)(a1 + 43) = 0;
    if ( !*(_DWORD *)(a2 + 240) )
      goto LABEL_6;
  }
  v15 = (_QWORD **)(*(_QWORD *)(a2 + 232) + 8 * v9);
  v13 = *v15;
  if ( *v15 )
  {
    *v15 = (_QWORD *)*v13;
    goto LABEL_12;
  }
LABEL_6:
  v11 = *(_QWORD *)(a2 + 128);
  v12 = 40LL << v10;
  *(_QWORD *)(a2 + 208) += 40LL << v10;
  v13 = (_QWORD *)((v11 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  v14 = (char *)v13 + (40LL << v10);
  if ( *(_QWORD *)(a2 + 136) >= (unsigned __int64)v14 && v11 )
    *(_QWORD *)(a2 + 128) = v14;
  else
    v13 = (_QWORD *)sub_9D1E70(a2 + 128, v12, v12, 3);
LABEL_12:
  *(_QWORD *)(a1 + 32) = v13;
  v16 = *(const __m128i **)(a3 + 32);
  v17 = (const __m128i *)((char *)v16 + 40 * (*(_DWORD *)(a3 + 40) & 0xFFFFFF));
  while ( v17 != v16 )
  {
    v18 = v16;
    v16 = (const __m128i *)((char *)v16 + 40);
    sub_2E8EAD0(a1, a2, v18);
  }
  if ( (*(_DWORD *)(a1 + 40) & 0xFFFFFF) != 0 )
  {
    v19 = 0;
    v20 = 40LL * (*(_DWORD *)(a1 + 40) & 0xFFFFFF);
    do
    {
      *(_WORD *)(*(_QWORD *)(a1 + 32) + v19 + 2) = *(_WORD *)(*(_QWORD *)(a3 + 32) + v19 + 2) & 0xFF0
                                                 | *(_WORD *)(*(_QWORD *)(a1 + 32) + v19 + 2) & 0xF00F;
      v19 += 40;
    }
    while ( v19 != v20 );
  }
  result = *(_DWORD *)(a3 + 44) & 0xFFFFF3 | *(_DWORD *)(a1 + 44) & 0xFF00000C;
  *(_DWORD *)(a1 + 44) = result;
  return result;
}
