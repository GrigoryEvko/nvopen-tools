// Function: sub_1381950
// Address: 0x1381950
//
__int64 __fastcall sub_1381950(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  int v6; // edx
  _QWORD *v8; // rax
  __int64 v9; // rdx
  _QWORD *v10; // r9
  __int64 v11; // rcx
  __int64 v12; // rdx
  _QWORD *v13; // rdx

  if ( *(_BYTE *)(a2 + 16) == 17 )
  {
    v6 = *(_DWORD *)(a2 + 32) + 1;
LABEL_3:
    *(_BYTE *)(a1 + 8) = 1;
    *(_DWORD *)a1 = v6;
    *(_DWORD *)(a1 + 4) = a3;
    return a1;
  }
  v8 = *(_QWORD **)a4;
  v9 = 8LL * *(unsigned int *)(a4 + 8);
  v10 = (_QWORD *)(*(_QWORD *)a4 + v9);
  v11 = v9 >> 3;
  v12 = v9 >> 5;
  if ( !v12 )
  {
LABEL_14:
    if ( v11 != 2 )
    {
      if ( v11 != 3 )
      {
        if ( v11 != 1 )
          goto LABEL_12;
        goto LABEL_17;
      }
      if ( a2 == *v8 )
        goto LABEL_11;
      ++v8;
    }
    if ( a2 == *v8 )
      goto LABEL_11;
    ++v8;
LABEL_17:
    if ( a2 != *v8 )
      goto LABEL_12;
    goto LABEL_11;
  }
  v13 = &v8[4 * v12];
  while ( a2 != *v8 )
  {
    if ( a2 == v8[1] )
    {
      ++v8;
      break;
    }
    if ( a2 == v8[2] )
    {
      v8 += 2;
      break;
    }
    if ( a2 == v8[3] )
    {
      v8 += 3;
      break;
    }
    v8 += 4;
    if ( v8 == v13 )
    {
      v11 = v10 - v8;
      goto LABEL_14;
    }
  }
LABEL_11:
  v6 = 0;
  if ( v10 != v8 )
    goto LABEL_3;
LABEL_12:
  *(_BYTE *)(a1 + 8) = 0;
  return a1;
}
