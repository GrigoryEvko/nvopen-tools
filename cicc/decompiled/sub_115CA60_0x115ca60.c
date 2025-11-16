// Function: sub_115CA60
// Address: 0x115ca60
//
__int64 __fastcall sub_115CA60(__int64 a1, int a2, unsigned __int8 *a3)
{
  __int64 result; // rax
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rax
  _QWORD *v9; // rcx
  __int64 v10; // rcx
  __int64 v11; // r12
  __int64 v12; // rdx
  _QWORD *v13; // rcx
  double *v14; // rdi
  __int64 v15; // rcx
  char v16; // al
  __int64 v17; // r12
  unsigned __int8 *v18; // [rsp-20h] [rbp-20h]
  unsigned __int8 *v19; // [rsp-20h] [rbp-20h]

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = *((_QWORD *)a3 - 8);
  v6 = *(_QWORD *)(v5 + 16);
  if ( v6
    && !*(_QWORD *)(v6 + 8)
    && *(_BYTE *)v5 == 86
    && ((*(_BYTE *)(v5 + 7) & 0x40) == 0
      ? (v13 = (_QWORD *)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF)))
      : (v13 = *(_QWORD **)(v5 - 8)),
        *v13
     && ((v14 = (double *)(a1 + 8), **(_QWORD **)a1 = *v13, (*(_BYTE *)(v5 + 7) & 0x40) == 0)
       ? (v15 = v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF))
       : (v15 = *(_QWORD *)(v5 - 8)),
         v19 = a3,
         v16 = sub_1009690(v14, *(_QWORD *)(v15 + 32)),
         a3 = v19,
         v16)) )
  {
    if ( (*(_BYTE *)(v5 + 7) & 0x40) != 0 )
      v17 = *(_QWORD *)(v5 - 8);
    else
      v17 = v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
    result = sub_1009690((double *)(a1 + 16), *(_QWORD *)(v17 + 64));
    a3 = v19;
    v7 = *((_QWORD *)v19 - 4);
    if ( (_BYTE)result && v7 )
    {
      **(_QWORD **)(a1 + 24) = v7;
      return result;
    }
  }
  else
  {
    v7 = *((_QWORD *)a3 - 4);
  }
  v8 = *(_QWORD *)(v7 + 16);
  if ( !v8 || *(_QWORD *)(v8 + 8) || *(_BYTE *)v7 != 86 )
    return 0;
  v9 = (*(_BYTE *)(v7 + 7) & 0x40) != 0
     ? *(_QWORD **)(v7 - 8)
     : (_QWORD *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF));
  if ( !*v9 )
    return 0;
  **(_QWORD **)a1 = *v9;
  v10 = (*(_BYTE *)(v7 + 7) & 0x40) != 0 ? *(_QWORD *)(v7 - 8) : v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
  v18 = a3;
  if ( !(unsigned __int8)sub_1009690((double *)(a1 + 8), *(_QWORD *)(v10 + 32)) )
    return 0;
  v11 = (*(_BYTE *)(v7 + 7) & 0x40) != 0 ? *(_QWORD *)(v7 - 8) : v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
  result = sub_1009690((double *)(a1 + 16), *(_QWORD *)(v11 + 64));
  if ( !(_BYTE)result )
    return 0;
  v12 = *((_QWORD *)v18 - 8);
  if ( !v12 )
    return 0;
  **(_QWORD **)(a1 + 24) = v12;
  return result;
}
