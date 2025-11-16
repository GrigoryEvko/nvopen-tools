// Function: sub_15D6A20
// Address: 0x15d6a20
//
__int64 __fastcall sub_15D6A20(__int64 a1, __int64 a2, __int64 *a3)
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
  _QWORD v20[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = sub_15D0890(a2, a3, v20);
  v7 = (__int64 *)v20[0];
  if ( v6 )
  {
    if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
    {
      v8 = a2 + 16;
      v9 = 64;
    }
    else
    {
      v8 = *(_QWORD *)(a2 + 16);
      v9 = 8LL * *(unsigned int *)(a2 + 24);
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
  if ( 4 * v15 >= v17 )
  {
    v16 *= 2;
    goto LABEL_19;
  }
  if ( v16 - (v15 + *(_DWORD *)(a2 + 12)) <= v16 >> 3 )
  {
LABEL_19:
    sub_15D6790(a2, v16);
    sub_15D0890(a2, a3, v20);
    v7 = (__int64 *)v20[0];
    v15 = (*(_DWORD *)(a2 + 8) >> 1) + 1;
  }
  *(_DWORD *)(a2 + 8) = *(_DWORD *)(a2 + 8) & 1 | (2 * v15);
  if ( *v7 != -8 )
    --*(_DWORD *)(a2 + 12);
  *v7 = *a3;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v18 = a2 + 16;
    v19 = 64;
  }
  else
  {
    v18 = *(_QWORD *)(a2 + 16);
    v19 = 8LL * *(unsigned int *)(a2 + 24);
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
