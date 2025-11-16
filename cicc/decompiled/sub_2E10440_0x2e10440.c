// Function: sub_2E10440
// Address: 0x2e10440
//
unsigned __int64 __fastcall sub_2E10440(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rsi
  __int64 v8; // rcx
  unsigned __int64 result; // rax
  unsigned int v10; // r11d
  __int64 v11; // rdi
  __int64 v12; // r9
  __int64 v13; // r10
  unsigned __int64 *v14; // r8
  unsigned __int64 v15; // rdx
  int v16; // eax
  __int64 v17; // rdx
  int v18; // ecx
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // r13
  int v24; // eax
  int v25; // ecx
  unsigned __int64 v26; // rcx
  unsigned __int64 v27; // r8
  int v28; // eax
  __int64 v29; // r14
  int v30; // eax

  v6 = a2;
  if ( **(_BYTE **)a1 )
    goto LABEL_2;
  v22 = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(v22 + 64) = 0;
  *(_DWORD *)(v22 + 8) = 0;
  v23 = *(_QWORD *)(a1 + 8);
  v24 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) + 16LL);
  v25 = *(_DWORD *)(v23 + 64) & 0x3F;
  if ( v25 )
    *(_QWORD *)(*(_QWORD *)v23 + 8LL * *(unsigned int *)(v23 + 8) - 8) |= -1LL << v25;
  v26 = *(unsigned int *)(v23 + 8);
  *(_DWORD *)(v23 + 64) = v24;
  v27 = (unsigned int)(v24 + 63) >> 6;
  if ( v27 != v26 )
  {
    if ( v27 >= v26 )
    {
      v29 = v27 - v26;
      if ( v27 > *(unsigned int *)(v23 + 12) )
      {
        sub_C8D5F0(v23, (const void *)(v23 + 16), v27, 8u, v27, a6);
        v26 = *(unsigned int *)(v23 + 8);
      }
      if ( 8 * v29 )
      {
        memset((void *)(*(_QWORD *)v23 + 8 * v26), 255, 8 * v29);
        LODWORD(v26) = *(_DWORD *)(v23 + 8);
      }
      v30 = *(_DWORD *)(v23 + 64);
      *(_DWORD *)(v23 + 8) = v29 + v26;
      v28 = v30 & 0x3F;
      if ( !v28 )
        goto LABEL_18;
      goto LABEL_17;
    }
    *(_DWORD *)(v23 + 8) = (unsigned int)(v24 + 63) >> 6;
  }
  v28 = v24 & 0x3F;
  if ( v28 )
LABEL_17:
    *(_QWORD *)(*(_QWORD *)v23 + 8LL * *(unsigned int *)(v23 + 8) - 8) &= ~(-1LL << v28);
LABEL_18:
  **(_BYTE **)a1 = 1;
LABEL_2:
  v7 = *(_QWORD *)(a1 + 8);
  v8 = *(_QWORD *)(**(_QWORD **)(a1 + 24) + 8 * v6);
  result = (unsigned int)(*(_DWORD *)(v7 + 64) + 31);
  v10 = (unsigned int)(*(_DWORD *)(v7 + 64) + 31) >> 5;
  if ( (unsigned int)result <= 0x3F )
  {
    LODWORD(v12) = 0;
  }
  else
  {
    v11 = 0;
    v12 = ((v10 - 2) >> 1) + 1;
    v13 = 8 * v12;
    do
    {
      v14 = (unsigned __int64 *)(v11 + *(_QWORD *)v7);
      v15 = *v14 & ~(unsigned __int64)(unsigned int)~*(_DWORD *)(v8 + v11);
      v16 = *(_DWORD *)(v8 + v11 + 4);
      v11 += 8;
      result = v15 & ~((unsigned __int64)(unsigned int)~v16 << 32);
      *v14 = result;
    }
    while ( v13 != v11 );
    v8 += v13;
    v10 &= 1u;
  }
  if ( v10 )
  {
    v17 = v8 + 4;
    v18 = 0;
    v19 = 8LL * (unsigned int)v12;
    v20 = v17;
    while ( 1 )
    {
      v21 = (unsigned __int64)(unsigned int)~*(_DWORD *)(v17 - 4) << v18;
      v18 += 32;
      result = ~v21;
      *(_QWORD *)(v19 + *(_QWORD *)v7) &= result;
      if ( v17 == v20 )
        break;
      v17 += 4;
    }
  }
  return result;
}
