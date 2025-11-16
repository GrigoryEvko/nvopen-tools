// Function: sub_F4FB10
// Address: 0xf4fb10
//
__int64 __fastcall sub_F4FB10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 result; // rax
  __int64 v14; // rbx
  __int64 v15; // rdx
  unsigned __int64 v16; // rcx

  v6 = a1;
  v9 = *(unsigned int *)(a2 + 8);
  if ( a1 )
  {
    v10 = v9 + 2;
    if ( v9 + 2 <= (unsigned __int64)*(unsigned int *)(a2 + 12) )
      goto LABEL_3;
    goto LABEL_12;
  }
  if ( v9 + 2 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, (const void *)(a2 + 16), v9 + 2, 8u, a5, a6);
    v9 = *(unsigned int *)(a2 + 8);
  }
  v15 = *(_QWORD *)a2;
  v6 = 1;
  *(_QWORD *)(v15 + 8 * v9) = 4101;
  *(_QWORD *)(v15 + 8 * v9 + 8) = 0;
  v16 = *(unsigned int *)(a2 + 12);
  v9 = (unsigned int)(*(_DWORD *)(a2 + 8) + 2);
  v10 = v9 + 2;
  *(_DWORD *)(a2 + 8) = v9;
  if ( v9 + 2 > v16 )
  {
LABEL_12:
    sub_C8D5F0(a2, (const void *)(a2 + 16), v10, 8u, a5, a6);
    v9 = *(unsigned int *)(a2 + 8);
  }
LABEL_3:
  v11 = *(_QWORD *)a2;
  *(_QWORD *)(v11 + 8 * v9) = 4101;
  *(_QWORD *)(v11 + 8 * v9 + 8) = v6;
  *(_DWORD *)(a2 + 8) += 2;
  if ( (*(_BYTE *)(a4 + 7) & 0x40) != 0 )
    v12 = *(_QWORD *)(a4 - 8);
  else
    v12 = a4 - 32LL * (*(_DWORD *)(a4 + 4) & 0x7FFFFFF);
  result = *(unsigned int *)(a3 + 8);
  v14 = *(_QWORD *)(v12 + 32);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), result + 1, 8u, a5, a6);
    result = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * result) = v14;
  ++*(_DWORD *)(a3 + 8);
  return result;
}
