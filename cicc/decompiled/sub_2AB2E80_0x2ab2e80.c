// Function: sub_2AB2E80
// Address: 0x2ab2e80
//
char __fastcall sub_2AB2E80(_QWORD **a1, __int64 a2)
{
  int v2; // r13d
  char v3; // r14
  __int64 v4; // r15
  int v5; // r12d
  char result; // al
  __int64 v7; // rdx
  __int64 v8; // rsi
  unsigned int v9; // edi
  __int64 v10; // r8
  int v11; // r11d
  unsigned int i; // edx
  __int64 v13; // rcx
  unsigned int v14; // edx
  int v15; // edx
  __int64 v16; // r8
  int v17; // ecx
  unsigned int v18; // edx
  __int64 v19; // rdi
  int v20; // r9d
  unsigned __int64 v21; // [rsp+0h] [rbp-50h]
  __int64 v22; // [rsp+8h] [rbp-48h]

  v2 = *(_DWORD *)a2;
  v3 = *(_BYTE *)(a2 + 4);
  v4 = (*a1)[5];
  v21 = *(_QWORD *)a2;
  v22 = *a1[1];
  v5 = sub_2AAA2B0(v4, v22, *(_DWORD *)a2, v3);
  result = 1;
  if ( v5 != 3 )
  {
    result = sub_2AB2DA0(v4, v22, v21);
    if ( result )
      return 0;
    v7 = (*a1)[5];
    v8 = *a1[1];
    v9 = *(_DWORD *)(v7 + 152);
    v10 = *(_QWORD *)(v7 + 136);
    if ( v9 )
    {
      v11 = 1;
      for ( i = (v9 - 1) & ((v3 == 0) + 37 * v2 - 1); ; i = (v9 - 1) & v14 )
      {
        v13 = v10 + 40LL * i;
        if ( v2 == *(_DWORD *)v13 && v3 == *(_BYTE *)(v13 + 4) )
          break;
        if ( *(_DWORD *)v13 == -1 && *(_BYTE *)(v13 + 4) )
          goto LABEL_19;
        v14 = v11 + i;
        ++v11;
      }
    }
    else
    {
LABEL_19:
      v13 = v10 + 40LL * v9;
    }
    v15 = *(_DWORD *)(v13 + 32);
    v16 = *(_QWORD *)(v13 + 16);
    if ( !v15 )
      return v5 != 5;
    v17 = v15 - 1;
    v18 = (v15 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v19 = *(_QWORD *)(v16 + 24LL * v18);
    if ( v8 != v19 )
    {
      v20 = 1;
      while ( v19 != -4096 )
      {
        v18 = v17 & (v20 + v18);
        v19 = *(_QWORD *)(v16 + 24LL * v18);
        if ( v8 == v19 )
          return result;
        ++v20;
      }
      return v5 != 5;
    }
  }
  return result;
}
