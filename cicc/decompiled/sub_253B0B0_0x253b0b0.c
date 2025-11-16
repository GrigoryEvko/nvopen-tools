// Function: sub_253B0B0
// Address: 0x253b0b0
//
__int64 __fastcall sub_253B0B0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  _QWORD *v10; // rax

  v6 = *a1;
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) == 0 )
  {
    v8 = *(unsigned int *)(v6 + 8);
    v7 = 0;
    v9 = v8 + 1;
    if ( v8 + 1 <= (unsigned __int64)*(unsigned int *)(v6 + 12) )
      goto LABEL_3;
LABEL_5:
    sub_C8D5F0(*a1, (const void *)(v6 + 16), v9, 0x10u, a5, a6);
    v8 = *(unsigned int *)(v6 + 8);
    goto LABEL_3;
  }
  v7 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v8 = *(unsigned int *)(v6 + 8);
  v9 = v8 + 1;
  if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(v6 + 12) )
    goto LABEL_5;
LABEL_3:
  v10 = (_QWORD *)(*(_QWORD *)v6 + 16 * v8);
  *v10 = v7;
  v10[1] = a2;
  ++*(_DWORD *)(v6 + 8);
  return 1;
}
