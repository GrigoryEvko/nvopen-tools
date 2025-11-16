// Function: sub_354BE50
// Address: 0x354be50
//
int *__fastcall sub_354BE50(__int64 a1, int *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  int v6; // r10d
  unsigned int v7; // ecx
  int *v8; // r12
  int *v9; // rax
  int v10; // edi
  int v12; // ecx
  int v13; // ecx
  int v14; // edx
  _DWORD *v15; // r12
  int *v16; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    v16 = 0;
LABEL_18:
    v4 *= 2;
    goto LABEL_19;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 1;
  v7 = (v4 - 1) & (37 * *a2);
  v8 = (int *)(v5 + 88LL * v7);
  v9 = 0;
  v10 = *v8;
  if ( *a2 == *v8 )
    return v8 + 2;
  while ( v10 != 0x7FFFFFFF )
  {
    if ( !v9 && v10 == 0x80000000 )
      v9 = v8;
    v7 = (v4 - 1) & (v6 + v7);
    v8 = (int *)(v5 + 88LL * v7);
    v10 = *v8;
    if ( *a2 == *v8 )
      return v8 + 2;
    ++v6;
  }
  v12 = *(_DWORD *)(a1 + 16);
  if ( !v9 )
    v9 = v8;
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  v16 = v9;
  if ( 4 * v13 >= 3 * v4 )
    goto LABEL_18;
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
LABEL_19:
    sub_354BAE0(a1, v4);
    sub_3546EF0(a1, a2, &v16);
    v13 = *(_DWORD *)(a1 + 16) + 1;
    v9 = v16;
  }
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v9 != 0x7FFFFFFF )
    --*(_DWORD *)(a1 + 20);
  v14 = *a2;
  *((_QWORD *)v9 + 1) = 0;
  v15 = v9 + 2;
  *((_QWORD *)v9 + 2) = 0;
  *v9 = v14;
  *((_QWORD *)v9 + 3) = 0;
  *((_QWORD *)v9 + 4) = 0;
  *((_QWORD *)v9 + 5) = 0;
  *((_QWORD *)v9 + 6) = 0;
  *((_QWORD *)v9 + 7) = 0;
  *((_QWORD *)v9 + 8) = 0;
  *((_QWORD *)v9 + 9) = 0;
  *((_QWORD *)v9 + 10) = 0;
  sub_3547BF0((__int64 *)v9 + 1, 0);
  return v15;
}
