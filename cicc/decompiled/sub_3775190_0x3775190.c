// Function: sub_3775190
// Address: 0x3775190
//
__int64 __fastcall sub_3775190(__int64 a1, const void *a2, __int64 a3, unsigned int a4, unsigned int a5, __m128i a6)
{
  int **v8; // rdx
  int *v9; // r11
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // r8
  _QWORD *v14; // rax
  int v15; // edx
  int v16; // edi
  _QWORD *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 result; // rax
  unsigned __int8 *v21; // rax
  int v22; // edx
  int v23; // edi
  unsigned __int8 *v24; // rdx
  __int64 v25; // rax

  v8 = *(int ***)a1;
  v9 = **(int ***)a1;
  if ( *v9 < 0 )
  {
    *v9 = a4;
    goto LABEL_8;
  }
  if ( a4 == *v9 )
  {
    *(_BYTE *)v8[1] = 1;
    if ( *(_BYTE *)v8[1] )
      goto LABEL_4;
LABEL_8:
    v10 = *(_QWORD *)(a1 + 56);
    v11 = 16LL * a4;
    v12 = a5;
    if ( *(_DWORD *)(*(_QWORD *)(v10 + v11) + 24LL) != 156 )
      goto LABEL_10;
    v13 = v10 + 16LL * a5;
    if ( *(_DWORD *)(*(_QWORD *)v13 + 24LL) != 156 )
      goto LABEL_10;
LABEL_14:
    v21 = sub_3774B80(
            *(unsigned int **)(a1 + 64),
            (__int64 *)(v10 + v11),
            (__int64 *)(v10 + 16 * v12),
            (__int64)a2,
            v13,
            a6);
    v23 = v22;
    v24 = v21;
    v25 = *(_QWORD *)(a1 + 8);
    *(_QWORD *)v25 = v24;
    *(_DWORD *)(v25 + 8) = v23;
    goto LABEL_11;
  }
  if ( !*(_BYTE *)v8[1] )
    goto LABEL_8;
LABEL_4:
  v10 = *(_QWORD *)(a1 + 48);
  v11 = 16LL * a4;
  v12 = a5;
  if ( *(_DWORD *)(*(_QWORD *)(v10 + v11) + 24LL) == 156 )
  {
    v13 = v10 + 16LL * a5;
    if ( *(_DWORD *)(*(_QWORD *)v13 + 24LL) == 156 )
      goto LABEL_14;
  }
LABEL_10:
  v14 = sub_33FCE10(
          *(_QWORD *)(a1 + 16),
          *(unsigned int *)(a1 + 24),
          *(_QWORD *)(a1 + 32),
          *(_QWORD *)(a1 + 40),
          *(_QWORD *)(v10 + 16LL * a4),
          *(_QWORD *)(v10 + 16LL * a4 + 8),
          a6,
          *(_QWORD *)(v10 + 16 * v12),
          *(_QWORD *)(v10 + 16 * v12 + 8),
          a2,
          a3);
  v16 = v15;
  v17 = v14;
  v18 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)v18 = v17;
  *(_DWORD *)(v18 + 8) = v16;
LABEL_11:
  v19 = *(_QWORD *)(a1 + 8);
  result = *(_QWORD *)(a1 + 48);
  *(_QWORD *)(result + v11) = *(_QWORD *)v19;
  *(_DWORD *)(result + v11 + 8) = *(_DWORD *)(v19 + 8);
  return result;
}
