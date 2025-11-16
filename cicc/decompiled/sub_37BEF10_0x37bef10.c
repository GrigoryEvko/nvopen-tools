// Function: sub_37BEF10
// Address: 0x37bef10
//
int *__fastcall sub_37BEF10(__int64 a1, int *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r9
  int v6; // r11d
  unsigned int v7; // r8d
  int *v8; // rdx
  int *v9; // rax
  int v10; // edi
  int *result; // rax
  int v12; // ecx
  int v13; // ecx
  int v14; // edx
  _DWORD *v15; // rdx
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
  v7 = (v4 - 1) & *a2;
  v8 = (int *)(v5 + 88LL * v7);
  v9 = 0;
  v10 = *v8;
  if ( *a2 == *v8 )
    return v8 + 2;
  while ( v10 != -1 )
  {
    if ( v10 == -2 && !v9 )
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
    sub_37BEC40(a1, v4);
    sub_37BDB10(a1, a2, &v16);
    v13 = *(_DWORD *)(a1 + 16) + 1;
    v9 = v16;
  }
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v9 != -1 )
    --*(_DWORD *)(a1 + 20);
  v14 = *a2;
  *((_QWORD *)v9 + 7) = 0;
  *(_OWORD *)(v9 + 10) = 0;
  *v9 = v14;
  *((_QWORD *)v9 + 1) = v9 + 6;
  v15 = v9 + 12;
  result = v9 + 2;
  *((_OWORD *)result + 1) = 0;
  *((_QWORD *)result + 1) = 0x400000000LL;
  result[10] = 0;
  *((_QWORD *)result + 7) = v15;
  *((_QWORD *)result + 8) = v15;
  *((_QWORD *)result + 9) = 0;
  return result;
}
