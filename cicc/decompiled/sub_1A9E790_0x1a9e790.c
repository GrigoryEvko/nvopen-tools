// Function: sub_1A9E790
// Address: 0x1a9e790
//
void __fastcall sub_1A9E790(__int64 a1, __int64 a2, __int64 a3)
{
  const void *v3; // r14
  __int64 v6; // rbx
  __int64 v7; // rdi
  int v8; // esi
  unsigned int v9; // edx
  __int64 *v10; // r10
  __int64 v11; // r9
  _QWORD *v12; // rax
  char v13; // cl
  __int64 v14; // rax
  int v15; // ecx
  unsigned int v16; // esi
  unsigned int v17; // edx
  __int64 *v18; // r8
  int v19; // edi
  int v20; // r8d
  int v21; // r9d
  __int64 v22; // r15
  __int64 v23; // rax
  int v24; // r11d
  __int64 v25; // [rsp-48h] [rbp-48h] BYREF
  __int64 *v26; // [rsp-40h] [rbp-40h] BYREF

  if ( a2 != a3 )
  {
    v3 = (const void *)(a1 + 288);
    v6 = a2;
    while ( 1 )
    {
      v12 = sub_1648700(v6);
      v13 = *(_BYTE *)(a1 + 8);
      v14 = v12[5];
      v25 = v14;
      v15 = v13 & 1;
      if ( v15 )
      {
        v7 = a1 + 16;
        v8 = 31;
      }
      else
      {
        v16 = *(_DWORD *)(a1 + 24);
        v7 = *(_QWORD *)(a1 + 16);
        if ( !v16 )
        {
          v17 = *(_DWORD *)(a1 + 8);
          ++*(_QWORD *)a1;
          v18 = 0;
          v19 = (v17 >> 1) + 1;
          goto LABEL_12;
        }
        v8 = v16 - 1;
      }
      v9 = v8 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v10 = (__int64 *)(v7 + 8LL * v9);
      v11 = *v10;
      if ( v14 != *v10 )
        break;
      do
LABEL_5:
        v6 = *(_QWORD *)(v6 + 8);
      while ( v6 && (unsigned __int8)(*((_BYTE *)sub_1648700(v6) + 16) - 25) > 9u );
      if ( a3 == v6 )
        return;
    }
    v24 = 1;
    v18 = 0;
    while ( v11 != -8 )
    {
      if ( v11 != -16 || v18 )
        v10 = v18;
      v9 = v8 & (v24 + v9);
      v11 = *(_QWORD *)(v7 + 8LL * v9);
      if ( v14 == v11 )
        goto LABEL_5;
      ++v24;
      v18 = v10;
      v10 = (__int64 *)(v7 + 8LL * v9);
    }
    v17 = *(_DWORD *)(a1 + 8);
    if ( !v18 )
      v18 = v10;
    ++*(_QWORD *)a1;
    v19 = (v17 >> 1) + 1;
    if ( (_BYTE)v15 )
    {
      v16 = 32;
      if ( (unsigned int)(4 * v19) >= 0x60 )
      {
LABEL_26:
        v16 *= 2;
        goto LABEL_27;
      }
    }
    else
    {
      v16 = *(_DWORD *)(a1 + 24);
LABEL_12:
      if ( 3 * v16 <= 4 * v19 )
        goto LABEL_26;
    }
    if ( v16 - *(_DWORD *)(a1 + 12) - v19 > v16 >> 3 )
    {
LABEL_14:
      *(_DWORD *)(a1 + 8) = (2 * (v17 >> 1) + 2) | v17 & 1;
      if ( *v18 != -8 )
        --*(_DWORD *)(a1 + 12);
      *v18 = v14;
      v22 = sub_1648700(v6)[5];
      v23 = *(unsigned int *)(a1 + 280);
      if ( (unsigned int)v23 >= *(_DWORD *)(a1 + 284) )
      {
        sub_16CD150(a1 + 272, v3, 0, 8, v20, v21);
        v23 = *(unsigned int *)(a1 + 280);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 272) + 8 * v23) = v22;
      ++*(_DWORD *)(a1 + 280);
      goto LABEL_5;
    }
LABEL_27:
    sub_1A9E4F0(a1, v16);
    sub_1A97A00(a1, &v25, &v26);
    v18 = v26;
    v14 = v25;
    v17 = *(_DWORD *)(a1 + 8);
    goto LABEL_14;
  }
}
