// Function: sub_15D6E30
// Address: 0x15d6e30
//
__int64 __fastcall sub_15D6E30(__int64 a1, __int64 a2, __int64 *a3, _DWORD *a4)
{
  char v8; // al
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rsi
  unsigned int v14; // eax
  int v15; // eax
  unsigned int v16; // esi
  unsigned int v17; // edi
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rsi
  _QWORD v21[7]; // [rsp+8h] [rbp-38h] BYREF

  v8 = sub_15D0950(a2, a3, v21);
  v9 = v21[0];
  if ( v8 )
  {
    if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
    {
      v10 = a2 + 16;
      v11 = 128;
    }
    else
    {
      v10 = *(_QWORD *)(a2 + 16);
      v11 = 16LL * *(unsigned int *)(a2 + 24);
    }
    v12 = *(_QWORD *)a2;
    *(_QWORD *)a1 = a2;
    *(_QWORD *)(a1 + 16) = v9;
    *(_QWORD *)(a1 + 8) = v12;
    *(_QWORD *)(a1 + 24) = v11 + v10;
    *(_BYTE *)(a1 + 32) = 0;
    return a1;
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
    goto LABEL_19;
  }
  if ( v16 - (v15 + *(_DWORD *)(a2 + 12)) <= v16 >> 3 )
  {
LABEL_19:
    sub_15D6B70(a2, v16);
    sub_15D0950(a2, a3, v21);
    v9 = v21[0];
    v15 = (*(_DWORD *)(a2 + 8) >> 1) + 1;
  }
  *(_DWORD *)(a2 + 8) = *(_DWORD *)(a2 + 8) & 1 | (2 * v15);
  if ( *(_QWORD *)v9 != -8 )
    --*(_DWORD *)(a2 + 12);
  *(_QWORD *)v9 = *a3;
  *(_DWORD *)(v9 + 8) = *a4;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v18 = a2 + 16;
    v19 = 128;
  }
  else
  {
    v18 = *(_QWORD *)(a2 + 16);
    v19 = 16LL * *(unsigned int *)(a2 + 24);
  }
  v20 = *(_QWORD *)a2;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v9;
  *(_QWORD *)(a1 + 8) = v20;
  *(_QWORD *)(a1 + 24) = v19 + v18;
  *(_BYTE *)(a1 + 32) = 1;
  return a1;
}
