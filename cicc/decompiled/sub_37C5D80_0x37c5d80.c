// Function: sub_37C5D80
// Address: 0x37c5d80
//
__int64 __fastcall sub_37C5D80(__int64 a1, __int64 a2, unsigned __int16 *a3, _DWORD *a4)
{
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  int v12; // esi
  __int16 *v13; // rax
  int v14; // edx
  unsigned int v15; // esi
  unsigned __int16 v16; // dx
  __int16 v17; // dx
  __int64 v18; // rcx
  __int16 *v19; // [rsp+0h] [rbp-30h] BYREF
  __int16 *v20; // [rsp+8h] [rbp-28h] BYREF

  if ( (unsigned __int8)sub_37BD660(a2, a3, &v19) )
  {
    v7 = *(unsigned int *)(a2 + 24);
    v8 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)a1 = a2;
    *(_BYTE *)(a1 + 32) = 0;
    v9 = v8 + 8 * v7;
    v10 = *(_QWORD *)a2;
    *(_QWORD *)(a1 + 24) = v9;
    *(_QWORD *)(a1 + 8) = v10;
    *(_QWORD *)(a1 + 16) = v19;
    return a1;
  }
  v12 = *(_DWORD *)(a2 + 16);
  v13 = v19;
  ++*(_QWORD *)a2;
  v14 = v12 + 1;
  v15 = *(_DWORD *)(a2 + 24);
  v20 = v13;
  if ( 4 * v14 >= 3 * v15 )
  {
    v15 *= 2;
  }
  else if ( v15 - *(_DWORD *)(a2 + 20) - v14 > v15 >> 3 )
  {
    goto LABEL_5;
  }
  sub_37C5A70(a2, v15);
  sub_37BD660(a2, a3, &v20);
  v14 = *(_DWORD *)(a2 + 16) + 1;
  v13 = v20;
LABEL_5:
  *(_DWORD *)(a2 + 16) = v14;
  if ( *v13 != -1 || v13[1] != -1 )
    --*(_DWORD *)(a2 + 20);
  v16 = *a3;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v13;
  *v13 = v16;
  v17 = a3[1];
  *(_BYTE *)(a1 + 32) = 1;
  v13[1] = v17;
  *((_DWORD *)v13 + 1) = *a4;
  v18 = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24);
  *(_QWORD *)(a1 + 8) = v18;
  return a1;
}
