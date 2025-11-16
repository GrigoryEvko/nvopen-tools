// Function: sub_2AAE960
// Address: 0x2aae960
//
__int64 __fastcall sub_2AAE960(_QWORD *a1, int *a2, __int64 a3, __int64 a4)
{
  _BYTE **v4; // rax
  _BYTE *v5; // r8
  char v6; // r10
  int v7; // r9d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdi
  int v11; // eax
  int v12; // ebx
  unsigned int i; // eax
  unsigned int *v14; // rsi
  unsigned int v15; // eax
  __int64 v17; // rax

  v4 = (_BYTE **)a1[1];
  v5 = *v4;
  if ( **v4 == 85 )
  {
    v17 = *((_QWORD *)v5 - 4);
    if ( v17 )
    {
      if ( !*(_BYTE *)v17
        && *(_QWORD *)(v17 + 24) == *((_QWORD *)v5 + 10)
        && (*(_BYTE *)(v17 + 33) & 0x20) != 0
        && *(_DWORD *)(v17 + 36) == 291 )
      {
        return 0;
      }
    }
  }
  v6 = *((_BYTE *)a2 + 4);
  v7 = *a2;
  v8 = *(_QWORD *)(*a1 + 40LL);
  if ( v6 )
  {
    v9 = *(unsigned int *)(v8 + 184);
    v10 = *(_QWORD *)(v8 + 168);
    if ( (_DWORD)v9 )
    {
      v11 = 37 * v7 - 1;
LABEL_6:
      v12 = 1;
      for ( i = (v9 - 1) & v11; ; i = (v9 - 1) & v15 )
      {
        v14 = (unsigned int *)(v10 + 72LL * i);
        a4 = *v14;
        if ( (_DWORD)a4 == v7 && v6 == *((_BYTE *)v14 + 4) )
        {
          v10 += 72LL * i;
          return sub_B19060(v10 + 8, (__int64)v5, v9, a4);
        }
        if ( (_DWORD)a4 == -1 && *((_BYTE *)v14 + 4) )
          break;
        v15 = v12 + i;
        ++v12;
      }
      v10 += 72 * v9;
    }
    return sub_B19060(v10 + 8, (__int64)v5, v9, a4);
  }
  if ( v7 != 1 )
  {
    v9 = *(unsigned int *)(v8 + 184);
    v10 = *(_QWORD *)(v8 + 168);
    if ( (_DWORD)v9 )
    {
      v11 = 37 * v7;
      goto LABEL_6;
    }
    return sub_B19060(v10 + 8, (__int64)v5, v9, a4);
  }
  return 1;
}
