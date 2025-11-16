// Function: sub_1AF5D70
// Address: 0x1af5d70
//
__int64 __fastcall sub_1AF5D70(__int64 a1, __int64 a2, __int64 *a3)
{
  char v6; // al
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // rax
  char v12; // cl
  unsigned int v14; // eax
  int v15; // eax
  unsigned int v16; // esi
  unsigned int v17; // edi
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 *v20[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = sub_1AF57C0(a2, a3, v20);
  v7 = v20[0];
  if ( v6 )
  {
    if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
    {
      v8 = a2 + 16;
      v9 = 192;
    }
    else
    {
      v8 = *(_QWORD *)(a2 + 16);
      v9 = 24LL * *(unsigned int *)(a2 + 24);
    }
    v10 = *(_QWORD *)a2;
    v11 = v9 + v8;
    v12 = 0;
    goto LABEL_5;
  }
  v14 = *(_DWORD *)(a2 + 8);
  ++*(_QWORD *)a2;
  v15 = (v14 >> 1) + 1;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v17 = 24;
    v16 = 8;
  }
  else
  {
    v16 = *(_DWORD *)(a2 + 24);
    v17 = 3 * v16;
  }
  if ( v17 <= 4 * v15 )
  {
    v16 *= 2;
  }
  else if ( v16 - (v15 + *(_DWORD *)(a2 + 12)) > v16 >> 3 )
  {
    goto LABEL_11;
  }
  sub_1AF5980(a2, v16);
  sub_1AF57C0(a2, a3, v20);
  v7 = v20[0];
  v15 = (*(_DWORD *)(a2 + 8) >> 1) + 1;
LABEL_11:
  *(_DWORD *)(a2 + 8) = *(_DWORD *)(a2 + 8) & 1 | (2 * v15);
  if ( *v7 != -8 || v7[1] != -8 || v7[2] != -8 )
    --*(_DWORD *)(a2 + 12);
  *v7 = *a3;
  v7[1] = a3[1];
  v7[2] = a3[2];
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v18 = a2 + 16;
    v19 = 192;
  }
  else
  {
    v18 = *(_QWORD *)(a2 + 16);
    v19 = 24LL * *(unsigned int *)(a2 + 24);
  }
  v11 = v19 + v18;
  v10 = *(_QWORD *)a2;
  v12 = 1;
LABEL_5:
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 24) = v11;
  *(_QWORD *)(a1 + 8) = v10;
  *(_QWORD *)(a1 + 16) = v7;
  *(_BYTE *)(a1 + 32) = v12;
  return a1;
}
