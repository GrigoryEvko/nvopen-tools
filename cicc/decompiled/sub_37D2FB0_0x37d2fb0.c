// Function: sub_37D2FB0
// Address: 0x37d2fb0
//
__int64 __fastcall sub_37D2FB0(__int64 a1, __int64 a2, __int64 *a3, _DWORD *a4)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v12; // rcx
  unsigned int v13; // eax
  __int64 v14; // rdx
  int v15; // eax
  unsigned int v16; // esi
  unsigned int v17; // edi
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v22[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( (unsigned __int8)sub_37BF550(a2, a3, &v21) )
  {
    if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
    {
      v8 = a2 + 16;
      v9 = 64;
    }
    else
    {
      v8 = *(_QWORD *)(a2 + 16);
      v9 = 16LL * *(unsigned int *)(a2 + 24);
    }
    v10 = *(_QWORD *)a2;
    *(_QWORD *)a1 = a2;
    *(_QWORD *)(a1 + 24) = v9 + v8;
    *(_QWORD *)(a1 + 8) = v10;
    v12 = v21;
    *(_BYTE *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 16) = v12;
    return a1;
  }
  v13 = *(_DWORD *)(a2 + 8);
  v14 = v21;
  ++*(_QWORD *)a2;
  v22[0] = v14;
  v15 = (v13 >> 1) + 1;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v17 = 12;
    v16 = 4;
  }
  else
  {
    v16 = *(_DWORD *)(a2 + 24);
    v17 = 3 * v16;
  }
  if ( 4 * v15 >= v17 )
  {
    v16 *= 2;
    goto LABEL_18;
  }
  if ( v16 - (v15 + *(_DWORD *)(a2 + 12)) <= v16 >> 3 )
  {
LABEL_18:
    sub_37D2C50(a2, v16);
    sub_37BF550(a2, a3, v22);
    v14 = v22[0];
    v15 = (*(_DWORD *)(a2 + 8) >> 1) + 1;
  }
  *(_DWORD *)(a2 + 8) = *(_DWORD *)(a2 + 8) & 1 | (2 * v15);
  if ( unk_5051170 != *(_QWORD *)v14 )
    --*(_DWORD *)(a2 + 12);
  *(_QWORD *)v14 = *a3;
  *(_DWORD *)(v14 + 8) = *a4;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v18 = a2 + 16;
    v19 = 64;
  }
  else
  {
    v18 = *(_QWORD *)(a2 + 16);
    v19 = 16LL * *(unsigned int *)(a2 + 24);
  }
  v20 = *(_QWORD *)a2;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 24) = v19 + v18;
  *(_QWORD *)(a1 + 8) = v20;
  *(_QWORD *)(a1 + 16) = v14;
  *(_BYTE *)(a1 + 32) = 1;
  return a1;
}
