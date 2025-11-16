// Function: sub_D0B410
// Address: 0xd0b410
//
__int64 __fastcall sub_D0B410(__int64 a1, __int64 a2, __m128i *a3, __int64 *a4)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 *v12; // rcx
  unsigned int v13; // eax
  __int64 *v14; // rdx
  int v15; // eax
  unsigned int v16; // esi
  unsigned int v17; // edi
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rsi
  __int64 *v21; // [rsp+0h] [rbp-30h] BYREF
  __int64 *v22; // [rsp+8h] [rbp-28h] BYREF

  if ( (unsigned __int8)sub_D0A160(a2, (unsigned __int64 *)a3, &v21) )
  {
    if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
    {
      v8 = a2 + 16;
      v9 = 320;
    }
    else
    {
      v8 = *(_QWORD *)(a2 + 16);
      v9 = 40LL * *(unsigned int *)(a2 + 24);
    }
    v10 = *(_QWORD *)a2;
    *(_QWORD *)a1 = a2;
    *(_QWORD *)(a1 + 24) = v9 + v8;
    *(_QWORD *)(a1 + 8) = v10;
    v12 = v21;
    *(_BYTE *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 16) = v12;
    return a1;
  }
  v13 = *(_DWORD *)(a2 + 8);
  v14 = v21;
  ++*(_QWORD *)a2;
  v22 = v14;
  v15 = (v13 >> 1) + 1;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v17 = 24;
    v16 = 8;
  }
  else
  {
    v16 = *(_DWORD *)(a2 + 24);
    v17 = 3 * v16;
  }
  if ( 4 * v15 >= v17 )
  {
    v16 *= 2;
  }
  else if ( v16 - (v15 + *(_DWORD *)(a2 + 12)) > v16 >> 3 )
  {
    goto LABEL_10;
  }
  sub_D0AF90(a2, v16);
  sub_D0A160(a2, (unsigned __int64 *)a3, &v22);
  v14 = v22;
  v15 = (*(_DWORD *)(a2 + 8) >> 1) + 1;
LABEL_10:
  *(_DWORD *)(a2 + 8) = *(_DWORD *)(a2 + 8) & 1 | (2 * v15);
  if ( *v14 != -4 || v14[1] != -3 || v14[2] != -4 || v14[3] != -3 )
    --*(_DWORD *)(a2 + 12);
  *(__m128i *)v14 = _mm_loadu_si128(a3);
  *((__m128i *)v14 + 1) = _mm_loadu_si128(a3 + 1);
  v14[4] = *a4;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v18 = a2 + 16;
    v19 = 320;
  }
  else
  {
    v18 = *(_QWORD *)(a2 + 16);
    v19 = 40LL * *(unsigned int *)(a2 + 24);
  }
  v20 = *(_QWORD *)a2;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 24) = v19 + v18;
  *(_QWORD *)(a1 + 8) = v20;
  *(_QWORD *)(a1 + 16) = v14;
  *(_BYTE *)(a1 + 32) = 1;
  return a1;
}
