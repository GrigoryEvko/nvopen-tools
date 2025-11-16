// Function: sub_26509C0
// Address: 0x26509c0
//
int *__fastcall sub_26509C0(__int64 a1, int *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r9
  int v6; // r11d
  int *v7; // rdi
  unsigned int v8; // ecx
  int *v9; // rax
  int v10; // r8d
  int v12; // eax
  int v13; // edx
  int v14; // eax
  int *v15; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    v15 = 0;
LABEL_18:
    v4 *= 2;
    goto LABEL_19;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 1;
  v7 = 0;
  v8 = (v4 - 1) & (37 * *a2);
  v9 = (int *)(v5 + 8LL * v8);
  v10 = *v9;
  if ( *a2 == *v9 )
    return v9 + 1;
  while ( v10 != -1 )
  {
    if ( v10 == -2 && !v7 )
      v7 = v9;
    v8 = (v4 - 1) & (v6 + v8);
    v9 = (int *)(v5 + 8LL * v8);
    v10 = *v9;
    if ( *a2 == *v9 )
      return v9 + 1;
    ++v6;
  }
  if ( !v7 )
    v7 = v9;
  v12 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  v15 = v7;
  if ( 4 * (v12 + 1) >= 3 * v4 )
    goto LABEL_18;
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
LABEL_19:
    sub_2650820(a1, v4);
    sub_264A480(a1, a2, &v15);
    v7 = v15;
    v13 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v7 != -1 )
    --*(_DWORD *)(a1 + 20);
  v14 = *a2;
  *((_BYTE *)v7 + 4) = 0;
  *v7 = v14;
  return v7 + 1;
}
