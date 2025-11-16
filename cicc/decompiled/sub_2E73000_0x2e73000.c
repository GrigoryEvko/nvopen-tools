// Function: sub_2E73000
// Address: 0x2e73000
//
__int64 __fastcall sub_2E73000(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 *v10; // rdx
  char v11; // cl
  unsigned int v13; // eax
  int v14; // eax
  unsigned int v15; // esi
  unsigned int v16; // edi
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 *v19; // [rsp+0h] [rbp-30h] BYREF
  __int64 *v20; // [rsp+8h] [rbp-28h] BYREF

  if ( (unsigned __int8)sub_2E6EF00(a2, a3, &v19) )
  {
    if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
    {
      v6 = a2 + 16;
      v7 = 64;
    }
    else
    {
      v6 = *(_QWORD *)(a2 + 16);
      v7 = 8LL * *(unsigned int *)(a2 + 24);
    }
    v8 = v7 + v6;
    v9 = *(_QWORD *)a2;
    v10 = v19;
    v11 = 0;
    goto LABEL_5;
  }
  v13 = *(_DWORD *)(a2 + 8);
  v10 = v19;
  ++*(_QWORD *)a2;
  v20 = v10;
  v14 = (v13 >> 1) + 1;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v16 = 24;
    v15 = 8;
  }
  else
  {
    v15 = *(_DWORD *)(a2 + 24);
    v16 = 3 * v15;
  }
  if ( 4 * v14 >= v16 )
  {
    v15 *= 2;
    goto LABEL_19;
  }
  if ( v15 - (v14 + *(_DWORD *)(a2 + 12)) <= v15 >> 3 )
  {
LABEL_19:
    sub_2E72D00(a2, v15);
    sub_2E6EF00(a2, a3, &v20);
    v10 = v20;
    v14 = (*(_DWORD *)(a2 + 8) >> 1) + 1;
  }
  *(_DWORD *)(a2 + 8) = *(_DWORD *)(a2 + 8) & 1 | (2 * v14);
  if ( *v10 != -4096 )
    --*(_DWORD *)(a2 + 12);
  *v10 = *a3;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v17 = a2 + 16;
    v18 = 64;
  }
  else
  {
    v17 = *(_QWORD *)(a2 + 16);
    v18 = 8LL * *(unsigned int *)(a2 + 24);
  }
  v8 = v18 + v17;
  v9 = *(_QWORD *)a2;
  v11 = 1;
LABEL_5:
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 24) = v8;
  *(_QWORD *)(a1 + 8) = v9;
  *(_QWORD *)(a1 + 16) = v10;
  *(_BYTE *)(a1 + 32) = v11;
  return a1;
}
