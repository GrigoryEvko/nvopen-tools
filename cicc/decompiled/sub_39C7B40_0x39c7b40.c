// Function: sub_39C7B40
// Address: 0x39c7b40
//
_QWORD *__fastcall sub_39C7B40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  _QWORD *result; // rax
  __int64 v11; // r14
  unsigned __int64 v12; // rax
  __int64 v13; // r14
  unsigned __int64 v14; // rax
  __int64 v15; // rdx

  v7 = *(_QWORD *)(a1 + 200);
  v8 = *(_QWORD *)(v7 + 4016);
  *(_QWORD *)(v7 + 4016) = a1;
  v9 = *(unsigned int *)(a1 + 816);
  if ( a1 != v8 || !(_DWORD)v9 )
    goto LABEL_3;
  v11 = *(_QWORD *)(*(_QWORD *)(a1 + 808) + 16 * v9 - 8);
  v12 = *(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v12
    || (*(_BYTE *)(v11 + 9) & 0xC) == 8
    && (*(_BYTE *)(v11 + 8) |= 4u,
        v12 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v11 + 24)),
        *(_QWORD *)v11 = v12 | *(_QWORD *)v11 & 7LL,
        v12) )
  {
    v13 = *(_QWORD *)(v12 + 24);
  }
  else
  {
    v13 = 0;
  }
  v14 = *(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v14
    || (*(_BYTE *)(a3 + 9) & 0xC) == 8
    && (*(_BYTE *)(a3 + 8) |= 4u,
        v14 = (unsigned __int64)sub_38CE440(*(_QWORD *)(a3 + 24)),
        *(_QWORD *)a3 = v14 | *(_QWORD *)a3 & 7LL,
        v14) )
  {
    v15 = *(_QWORD *)(v14 + 24);
  }
  else
  {
    v15 = 0;
  }
  v9 = *(unsigned int *)(a1 + 816);
  if ( v15 == v13 )
  {
    result = (_QWORD *)(*(_QWORD *)(a1 + 808) + 16 * v9);
    *(result - 1) = a3;
  }
  else
  {
LABEL_3:
    if ( *(_DWORD *)(a1 + 820) <= (unsigned int)v9 )
    {
      sub_16CD150(a1 + 808, (const void *)(a1 + 824), 0, 16, a5, a6);
      v9 = *(unsigned int *)(a1 + 816);
    }
    result = (_QWORD *)(*(_QWORD *)(a1 + 808) + 16 * v9);
    *result = a2;
    result[1] = a3;
    ++*(_DWORD *)(a1 + 816);
  }
  return result;
}
