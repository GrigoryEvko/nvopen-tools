// Function: sub_2D08010
// Address: 0x2d08010
//
__int64 __fastcall sub_2D08010(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r8d
  __int64 v5; // rdi
  unsigned int v6; // ecx
  __int64 *v7; // rax
  __int64 v8; // r10
  unsigned int v9; // r8d
  __int64 v10; // r12
  __int64 v11; // rcx
  unsigned int v12; // r9d
  int v13; // r11d
  unsigned int v14; // edx
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // r10
  unsigned int v19; // eax
  char v20; // cl
  __int64 v21; // rax
  int v23; // eax
  int v24; // r11d
  int v25; // r11d
  __int64 v26; // rsi
  int v27; // eax
  int v28; // edx
  int v29; // esi
  __int64 v30; // [rsp+0h] [rbp-30h] BYREF
  __int64 v31[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 136);
  v5 = *(_QWORD *)(a1 + 120);
  if ( v4 )
  {
    v6 = (v4 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( a3 == *v7 )
      goto LABEL_3;
    v23 = 1;
    while ( v8 != -4096 )
    {
      v24 = v23 + 1;
      v6 = (v4 - 1) & (v23 + v6);
      v7 = (__int64 *)(v5 + 16LL * v6);
      v8 = *v7;
      if ( a3 == *v7 )
        goto LABEL_3;
      v23 = v24;
    }
  }
  v7 = (__int64 *)(v5 + 16LL * v4);
LABEL_3:
  v9 = *(_DWORD *)(a1 + 80);
  v10 = v7[1];
  v30 = a2;
  v11 = *(_QWORD *)(a1 + 64);
  if ( !v9 )
    return 0;
  v12 = v9 - 1;
  v13 = 1;
  v14 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  LODWORD(v15) = v14;
  v16 = v11 + 16LL * v14;
  v17 = *(_QWORD *)v16;
  v18 = *(_QWORD *)v16;
  if ( a2 != *(_QWORD *)v16 )
  {
    while ( 1 )
    {
      if ( v18 == -4096 )
        return 0;
      v15 = v12 & ((_DWORD)v15 + v13);
      v18 = *(_QWORD *)(v11 + 16 * v15);
      if ( a2 == v18 )
        break;
      ++v13;
    }
    v25 = 1;
    v26 = 0;
    while ( v17 != -4096 )
    {
      if ( v17 == -8192 && !v26 )
        v26 = v16;
      v14 = v12 & (v25 + v14);
      v16 = v11 + 16LL * v14;
      v17 = *(_QWORD *)v16;
      if ( v18 == *(_QWORD *)v16 )
        goto LABEL_5;
      ++v25;
    }
    if ( !v26 )
      v26 = v16;
    v27 = *(_DWORD *)(a1 + 72);
    ++*(_QWORD *)(a1 + 56);
    v28 = v27 + 1;
    v31[0] = v26;
    if ( 4 * (v27 + 1) >= 3 * v9 )
    {
      v29 = 2 * v9;
    }
    else
    {
      if ( v9 - *(_DWORD *)(a1 + 76) - v28 > v9 >> 3 )
      {
LABEL_22:
        *(_DWORD *)(a1 + 72) = v28;
        if ( *(_QWORD *)v26 != -4096 )
          --*(_DWORD *)(a1 + 76);
        *(_QWORD *)v26 = v18;
        v20 = 0;
        v21 = 0;
        *(_DWORD *)(v26 + 8) = 0;
        return (*(_QWORD *)(*(_QWORD *)(v10 + 24) + v21) >> v20) & 1LL;
      }
      v29 = v9;
    }
    sub_CE2410(a1 + 56, v29);
    sub_2D064D0(a1 + 56, &v30, v31);
    v18 = v30;
    v26 = v31[0];
    v28 = *(_DWORD *)(a1 + 72) + 1;
    goto LABEL_22;
  }
LABEL_5:
  v19 = *(_DWORD *)(v16 + 8);
  v20 = v19 & 0x3F;
  v21 = 8LL * (v19 >> 6);
  return (*(_QWORD *)(*(_QWORD *)(v10 + 24) + v21) >> v20) & 1LL;
}
