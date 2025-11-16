// Function: sub_1E4A590
// Address: 0x1e4a590
//
int *__fastcall sub_1E4A590(__int64 a1, int *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  unsigned int v6; // edx
  int *v7; // r12
  int v8; // ecx
  int v10; // r10d
  int *v11; // rdi
  int v12; // eax
  int v13; // edx
  int v14; // eax
  _QWORD v15[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
LABEL_14:
    v4 *= 2;
    goto LABEL_15;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (v4 - 1) & (37 * *a2);
  v7 = (int *)(v5 + 88LL * v6);
  v8 = *v7;
  if ( *a2 == *v7 )
    return v7;
  v10 = 1;
  v11 = 0;
  while ( v8 != 0x7FFFFFFF )
  {
    if ( v8 == 0x80000000 && !v11 )
      v11 = v7;
    v6 = (v4 - 1) & (v10 + v6);
    v7 = (int *)(v5 + 88LL * v6);
    v8 = *v7;
    if ( *a2 == *v7 )
      return v7;
    ++v10;
  }
  v12 = *(_DWORD *)(a1 + 16);
  if ( v11 )
    v7 = v11;
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v4 )
    goto LABEL_14;
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
LABEL_15:
    sub_1E4A230(a1, v4);
    sub_1E48470(a1, a2, v15);
    v7 = (int *)v15[0];
    v13 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v7 != 0x7FFFFFFF )
    --*(_DWORD *)(a1 + 20);
  v14 = *a2;
  *((_QWORD *)v7 + 1) = 0;
  *((_QWORD *)v7 + 2) = 0;
  *v7 = v14;
  *((_QWORD *)v7 + 3) = 0;
  *((_QWORD *)v7 + 4) = 0;
  *((_QWORD *)v7 + 5) = 0;
  *((_QWORD *)v7 + 6) = 0;
  *((_QWORD *)v7 + 7) = 0;
  *((_QWORD *)v7 + 8) = 0;
  *((_QWORD *)v7 + 9) = 0;
  *((_QWORD *)v7 + 10) = 0;
  sub_1E47CF0((__int64 *)v7 + 1, 0);
  return v7;
}
