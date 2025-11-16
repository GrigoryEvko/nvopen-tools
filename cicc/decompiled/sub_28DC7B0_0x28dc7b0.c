// Function: sub_28DC7B0
// Address: 0x28dc7b0
//
__int64 __fastcall sub_28DC7B0(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  int v10; // esi
  __int64 v11; // rax
  int v12; // edx
  unsigned int v13; // esi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v17[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( (unsigned __int8)sub_28CE930(a2, (__int64 *)a3, &v16) )
  {
    v7 = *(unsigned int *)(a2 + 24);
    *(_QWORD *)a1 = a2;
    *(_BYTE *)(a1 + 32) = 0;
    v8 = *(_QWORD *)a2;
    *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 8) + 24 * v7;
    *(_QWORD *)(a1 + 8) = v8;
    *(_QWORD *)(a1 + 16) = v16;
    return a1;
  }
  v10 = *(_DWORD *)(a2 + 16);
  v11 = v16;
  ++*(_QWORD *)a2;
  v12 = v10 + 1;
  v13 = *(_DWORD *)(a2 + 24);
  v17[0] = v11;
  if ( 4 * v12 >= 3 * v13 )
  {
    v13 *= 2;
  }
  else if ( v13 - *(_DWORD *)(a2 + 20) - v12 > v13 >> 3 )
  {
    goto LABEL_5;
  }
  sub_28DC520(a2, v13);
  sub_28CE930(a2, (__int64 *)a3, v17);
  v12 = *(_DWORD *)(a2 + 16) + 1;
  v11 = v17[0];
LABEL_5:
  *(_DWORD *)(a2 + 16) = v12;
  if ( *(_QWORD *)v11 != -4096 || *(_DWORD *)(v11 + 8) != -1 )
    --*(_DWORD *)(a2 + 20);
  v14 = *(_QWORD *)a3;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v11;
  *(_QWORD *)v11 = v14;
  LODWORD(v14) = *(_DWORD *)(a3 + 8);
  *(_BYTE *)(a1 + 32) = 1;
  *(_DWORD *)(v11 + 8) = v14;
  *(_BYTE *)(v11 + 16) = *a4;
  v15 = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 8) + 24LL * *(unsigned int *)(a2 + 24);
  *(_QWORD *)(a1 + 8) = v15;
  return a1;
}
