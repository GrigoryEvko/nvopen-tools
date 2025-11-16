// Function: sub_10B5AF0
// Address: 0x10b5af0
//
__int64 __fastcall sub_10B5AF0(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // rax
  __int64 result; // rax
  int v4; // edx
  unsigned __int8 *v5; // rdx
  int v6; // eax
  unsigned __int8 *v7; // rdx
  __int64 v8; // rax
  unsigned __int8 *v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rcx

  v2 = *((_QWORD *)a2 + 2);
  if ( !v2 || *(_QWORD *)(v2 + 8) )
    return 0;
  v4 = *a2;
  if ( v4 != *(_DWORD *)(a1 + 32) + 29 )
    goto LABEL_5;
  v9 = (unsigned __int8 *)*((_QWORD *)a2 - 8);
  if ( *v9 != *(_DWORD *)(a1 + 16) + 29 )
    goto LABEL_5;
  v10 = *((_QWORD *)v9 - 8);
  if ( !v10 )
    goto LABEL_5;
  **(_QWORD **)a1 = v10;
  v11 = *((_QWORD *)v9 - 4);
  if ( !v11
    || (**(_QWORD **)(a1 + 8) = v11, result = sub_1009690((double *)(a1 + 24), *((_QWORD *)a2 - 4)), !(_BYTE)result) )
  {
    v12 = *((_QWORD *)a2 + 2);
    if ( !v12 || *(_QWORD *)(v12 + 8) )
      return 0;
    v4 = *a2;
LABEL_5:
    if ( v4 != *(_DWORD *)(a1 + 72) + 29 )
      return 0;
    v5 = (unsigned __int8 *)*((_QWORD *)a2 - 8);
    v6 = *(_DWORD *)(a1 + 56) + 29;
    if ( *v5 == v6 && (v13 = *((_QWORD *)v5 - 8)) != 0 )
    {
      **(_QWORD **)(a1 + 40) = v13;
      result = sub_1009690((double *)(a1 + 48), *((_QWORD *)v5 - 4));
      v7 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
      if ( (_BYTE)result && v7 )
        goto LABEL_14;
      v6 = *(_DWORD *)(a1 + 56) + 29;
    }
    else
    {
      v7 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
    }
    if ( *v7 != v6 )
      return 0;
    v8 = *((_QWORD *)v7 - 8);
    if ( !v8 )
      return 0;
    **(_QWORD **)(a1 + 40) = v8;
    result = sub_1009690((double *)(a1 + 48), *((_QWORD *)v7 - 4));
    if ( !(_BYTE)result )
      return 0;
    v7 = (unsigned __int8 *)*((_QWORD *)a2 - 8);
    if ( !v7 )
      return 0;
LABEL_14:
    **(_QWORD **)(a1 + 64) = v7;
  }
  return result;
}
