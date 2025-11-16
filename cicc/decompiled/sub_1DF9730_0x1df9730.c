// Function: sub_1DF9730
// Address: 0x1df9730
//
__int64 __fastcall sub_1DF9730(__int64 a1, int a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // esi
  int v5; // eax
  unsigned int v7; // r9d
  __int64 v8; // r8
  unsigned int v10; // edx
  int *v11; // rcx
  int v12; // edi
  unsigned int v13; // eax
  char v14; // cl
  __int64 v15; // rax
  int v16; // r10d
  unsigned int v17; // r11d
  int i; // r13d
  int v19; // r11d
  int *v20; // r10
  int v21; // edi
  int v22; // ecx
  int *v23; // r13
  unsigned int v24; // eax
  int v25[3]; // [rsp+Ch] [rbp-34h] BYREF
  int *v26; // [rsp+18h] [rbp-28h] BYREF

  result = 0;
  v25[0] = a2;
  v4 = *(_DWORD *)(a1 + 80);
  if ( v4 )
  {
    v5 = v25[0];
    v7 = v4 - 1;
    v8 = *(_QWORD *)(a1 + 64);
    v10 = (v4 - 1) & (37 * v25[0]);
    v11 = (int *)(v8 + 8LL * v10);
    v12 = *v11;
    if ( v25[0] == *v11 )
    {
      v13 = v11[1];
      v14 = v13 & 0x3F;
      v15 = 8LL * (v13 >> 6);
      return (*(_QWORD *)(*(_QWORD *)(a3 + 24) + v15) >> v14) & 1LL;
    }
    v16 = *v11;
    v17 = (v4 - 1) & (37 * v25[0]);
    for ( i = 1; ; ++i )
    {
      if ( v16 == -1 )
        return 0;
      v17 = v7 & (i + v17);
      v16 = *(_DWORD *)(v8 + 8LL * v17);
      if ( v25[0] == v16 )
        break;
    }
    v19 = 1;
    v20 = 0;
    while ( v12 != -1 )
    {
      if ( v20 || v12 != -2 )
        v11 = v20;
      v10 = v7 & (v19 + v10);
      v23 = (int *)(v8 + 8LL * v10);
      v12 = *v23;
      if ( v25[0] == *v23 )
      {
        v24 = v23[1];
        v14 = v24 & 0x3F;
        v15 = 8LL * (v24 >> 6);
        return (*(_QWORD *)(*(_QWORD *)(a3 + 24) + v15) >> v14) & 1LL;
      }
      ++v19;
      v20 = v11;
      v11 = (int *)(v8 + 8LL * v10);
    }
    v21 = *(_DWORD *)(a1 + 72);
    if ( !v20 )
      v20 = v11;
    ++*(_QWORD *)(a1 + 56);
    v22 = v21 + 1;
    if ( 4 * (v21 + 1) >= 3 * v4 )
    {
      v4 *= 2;
    }
    else if ( v4 - *(_DWORD *)(a1 + 76) - v22 > v4 >> 3 )
    {
LABEL_16:
      *(_DWORD *)(a1 + 72) = v22;
      if ( *v20 != -1 )
        --*(_DWORD *)(a1 + 76);
      *v20 = v5;
      v14 = 0;
      v15 = 0;
      v20[1] = 0;
      return (*(_QWORD *)(*(_QWORD *)(a3 + 24) + v15) >> v14) & 1LL;
    }
    sub_1BFDD60(a1 + 56, v4);
    sub_1BFD720(a1 + 56, v25, &v26);
    v20 = v26;
    v5 = v25[0];
    v22 = *(_DWORD *)(a1 + 72) + 1;
    goto LABEL_16;
  }
  return result;
}
