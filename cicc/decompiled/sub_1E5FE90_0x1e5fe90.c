// Function: sub_1E5FE90
// Address: 0x1e5fe90
//
char *__fastcall sub_1E5FE90(__int64 a1)
{
  __int64 v1; // r13
  unsigned int v3; // esi
  char *result; // rax
  __int64 v5; // rdx
  _BYTE *v6; // rsi
  char *v7; // rdi
  int v8; // r9d
  char *v9; // rcx
  unsigned int v10; // r8d
  bool v11; // zf
  int v12; // ecx
  int v13; // ecx
  __int64 v14; // rdi
  __int64 v15; // [rsp+0h] [rbp-20h] BYREF
  _QWORD v16[3]; // [rsp+8h] [rbp-18h] BYREF

  v1 = a1 + 24;
  v3 = *(_DWORD *)(a1 + 48);
  v15 = 0;
  if ( !v3 )
  {
    ++*(_QWORD *)(a1 + 24);
LABEL_18:
    v3 *= 2;
LABEL_19:
    sub_1E5FC50(v1, v3);
    sub_1E5FA10(v1, &v15, v16);
    result = (char *)v16[0];
    v14 = v15;
    v13 = *(_DWORD *)(a1 + 40) + 1;
    goto LABEL_14;
  }
  result = *(char **)(a1 + 32);
  v5 = *(_QWORD *)result;
  if ( !*(_QWORD *)result )
    goto LABEL_3;
  v7 = *(char **)(a1 + 32);
  v8 = 1;
  v9 = 0;
  v10 = 0;
  while ( v5 != -8 )
  {
    if ( v9 || v5 != -16 )
      v7 = v9;
    v10 = (v3 - 1) & (v8 + v10);
    v5 = *(_QWORD *)&result[72 * v10];
    if ( !v5 )
    {
      result += 72 * v10;
      goto LABEL_3;
    }
    ++v8;
    v9 = v7;
    v7 = &result[72 * v10];
  }
  v11 = v9 == 0;
  result = v9;
  v12 = *(_DWORD *)(a1 + 40);
  if ( v11 )
    result = v7;
  ++*(_QWORD *)(a1 + 24);
  v13 = v12 + 1;
  if ( 4 * v13 >= 3 * v3 )
    goto LABEL_18;
  v14 = 0;
  if ( v3 - *(_DWORD *)(a1 + 44) - v13 <= v3 >> 3 )
    goto LABEL_19;
LABEL_14:
  *(_DWORD *)(a1 + 40) = v13;
  if ( *(_QWORD *)result != -8 )
    --*(_DWORD *)(a1 + 44);
  *(_QWORD *)result = v14;
  *((_QWORD *)result + 5) = result + 56;
  *((_QWORD *)result + 6) = 0x200000000LL;
  *(_OWORD *)(result + 8) = 0;
  *(_OWORD *)(result + 24) = 0;
  *(_OWORD *)(result + 56) = 0;
LABEL_3:
  *((_DWORD *)result + 4) = 1;
  *((_DWORD *)result + 2) = 1;
  *((_QWORD *)result + 3) = 0;
  v6 = *(_BYTE **)(a1 + 8);
  v16[0] = 0;
  if ( v6 == *(_BYTE **)(a1 + 16) )
    return sub_1E5FAC0(a1, v6, v16);
  if ( v6 )
  {
    *(_QWORD *)v6 = 0;
    v6 = *(_BYTE **)(a1 + 8);
  }
  *(_QWORD *)(a1 + 8) = v6 + 8;
  return result;
}
