// Function: sub_37C05B0
// Address: 0x37c05b0
//
_QWORD *__fastcall sub_37C05B0(__int64 a1, __int64 *a2)
{
  char v4; // cl
  __int64 v5; // rdi
  int v6; // esi
  unsigned int v7; // edx
  _QWORD *v8; // rax
  __int64 v9; // r9
  _QWORD *result; // rax
  unsigned int v11; // esi
  unsigned int v12; // eax
  int v13; // edx
  unsigned int v14; // edi
  _QWORD *v15; // rax
  __int64 v16; // rdx
  int v17; // r11d
  _QWORD *v18; // r10
  _QWORD *v19; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( v4 )
  {
    v5 = a1 + 16;
    v6 = 15;
  }
  else
  {
    v11 = *(_DWORD *)(a1 + 24);
    v5 = *(_QWORD *)(a1 + 16);
    if ( !v11 )
    {
      v12 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v19 = 0;
      v13 = (v12 >> 1) + 1;
LABEL_8:
      v14 = 3 * v11;
      goto LABEL_9;
    }
    v6 = v11 - 1;
  }
  v7 = v6 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v8 = (_QWORD *)(v5 + 16LL * v7);
  v9 = *v8;
  if ( *a2 == *v8 )
    return v8 + 1;
  v17 = 1;
  v18 = 0;
  while ( v9 != -4096 )
  {
    if ( !v18 && v9 == -8192 )
      v18 = v8;
    v7 = v6 & (v17 + v7);
    v8 = (_QWORD *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( *a2 == *v8 )
      return v8 + 1;
    ++v17;
  }
  v14 = 48;
  v11 = 16;
  if ( !v18 )
    v18 = v8;
  v12 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v19 = v18;
  v13 = (v12 >> 1) + 1;
  if ( !v4 )
  {
    v11 = *(_DWORD *)(a1 + 24);
    goto LABEL_8;
  }
LABEL_9:
  if ( 4 * v13 >= v14 )
  {
    v11 *= 2;
    goto LABEL_15;
  }
  if ( v11 - *(_DWORD *)(a1 + 12) - v13 <= v11 >> 3 )
  {
LABEL_15:
    sub_37C0190(a1, v11);
    sub_37BE670(a1, a2, &v19);
    v12 = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = (2 * (v12 >> 1) + 2) | v12 & 1;
  v15 = v19;
  if ( *v19 != -4096 )
    --*(_DWORD *)(a1 + 12);
  v16 = *a2;
  v15[1] = 0;
  result = v15 + 1;
  *(result - 1) = v16;
  return result;
}
