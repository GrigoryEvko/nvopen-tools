// Function: sub_25B0BB0
// Address: 0x25b0bb0
//
__int64 __fastcall sub_25B0BB0(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v4; // r8
  unsigned __int64 v5; // rdx
  unsigned int v6; // ecx
  unsigned int v7; // edi
  __int64 v8; // rcx
  __int64 v9; // rsi

  result = *(_QWORD *)(a1 + 16);
  v4 = a1 + 8;
  if ( !result )
    return v4;
  v5 = *a2;
  while ( 1 )
  {
    if ( *(_QWORD *)(result + 32) < v5 )
      goto LABEL_8;
    if ( *(_QWORD *)(result + 32) != v5 )
      break;
    v6 = *(_DWORD *)(result + 40);
    v7 = *((_DWORD *)a2 + 2);
    if ( v6 >= v7 && (v6 != v7 || *(_BYTE *)(result + 44) >= *((_BYTE *)a2 + 12)) )
      goto LABEL_14;
LABEL_8:
    result = *(_QWORD *)(result + 24);
LABEL_9:
    if ( !result )
      return v4;
  }
  if ( *(_QWORD *)(result + 32) > v5 )
    goto LABEL_17;
  v6 = *(_DWORD *)(result + 40);
  v7 = *((_DWORD *)a2 + 2);
LABEL_14:
  if ( v7 < v6 || v7 == v6 && *((_BYTE *)a2 + 12) < *(_BYTE *)(result + 44) )
  {
LABEL_17:
    v4 = result;
    result = *(_QWORD *)(result + 16);
    goto LABEL_9;
  }
  v8 = *(_QWORD *)(result + 24);
  v9 = *(_QWORD *)(result + 16);
  while ( v8 )
  {
    if ( *(_QWORD *)(v8 + 32) > v5
      || *(_QWORD *)(v8 + 32) == v5
      && (v7 < *(_DWORD *)(v8 + 40) || v7 == *(_DWORD *)(v8 + 40) && *((_BYTE *)a2 + 12) < *(_BYTE *)(v8 + 44)) )
    {
      v8 = *(_QWORD *)(v8 + 16);
    }
    else
    {
      v8 = *(_QWORD *)(v8 + 24);
    }
  }
  while ( v9 )
  {
    if ( v5 > *(_QWORD *)(v9 + 32)
      || v5 == *(_QWORD *)(v9 + 32)
      && (v7 > *(_DWORD *)(v9 + 40) || v7 == *(_DWORD *)(v9 + 40) && *(_BYTE *)(v9 + 44) < *((_BYTE *)a2 + 12)) )
    {
      v9 = *(_QWORD *)(v9 + 24);
    }
    else
    {
      result = v9;
      v9 = *(_QWORD *)(v9 + 16);
    }
  }
  return result;
}
