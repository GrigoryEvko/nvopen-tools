// Function: sub_16BD4E0
// Address: 0x16bd4e0
//
unsigned __int64 __fastcall sub_16BD4E0(__int64 a1, unsigned __int8 *a2, unsigned int a3)
{
  unsigned __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rdx
  unsigned int v7; // r12d
  int v8; // ebx
  unsigned int v9; // r12d
  int v10; // edx
  int v11; // eax
  int v12; // ebx
  unsigned __int64 v13; // rbx
  size_t v14; // r8
  unsigned int v15; // [rsp+4h] [rbp-3Ch]

  result = a3;
  v5 = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)v5 >= *(_DWORD *)(a1 + 12) )
  {
    sub_16CD150(a1, a1 + 16, 0, 4);
    v5 = *(unsigned int *)(a1 + 8);
    result = a3;
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v5) = a3;
  v6 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v6;
  if ( a3 )
  {
    if ( ((unsigned __int8)a2 & 3) != 0 )
    {
      v7 = 4;
      if ( a3 > 3 )
      {
        do
        {
          v8 = (a2[v7 - 3] << 8) | a2[v7 - 4] | (a2[v7 - 2] << 16) | (a2[v7 - 1] << 24);
          if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v6 )
          {
            v15 = result;
            sub_16CD150(a1, a1 + 16, 0, 4);
            v6 = *(unsigned int *)(a1 + 8);
            result = v15;
          }
          v7 += 4;
          *(_DWORD *)(*(_QWORD *)a1 + 4 * v6) = v8;
          v6 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
          *(_DWORD *)(a1 + 8) = v6;
        }
        while ( (unsigned int)result >= v7 );
      }
      v9 = v7 - a3;
      if ( v9 != 2 )
      {
LABEL_10:
        if ( v9 == 3 )
        {
          v11 = 0;
          goto LABEL_14;
        }
        if ( v9 != 1 )
          return result;
        v10 = a2[a3 - 3] << 8;
LABEL_13:
        v11 = (v10 | a2[a3 - 2]) << 8;
LABEL_14:
        v12 = v11 | a2[a3 - 1];
        result = *(unsigned int *)(a1 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a1 + 12) )
        {
          sub_16CD150(a1, a1 + 16, 0, 4);
          result = *(unsigned int *)(a1 + 8);
        }
        *(_DWORD *)(*(_QWORD *)a1 + 4 * result) = v12;
        ++*(_DWORD *)(a1 + 8);
        return result;
      }
    }
    else
    {
      v13 = a3 >> 2;
      result = *(unsigned int *)(a1 + 12) - v6;
      v14 = 4 * v13;
      if ( v13 > result )
      {
        result = sub_16CD150(a1, a1 + 16, v13 + v6, 4);
        v6 = *(unsigned int *)(a1 + 8);
        v14 = 4 * v13;
      }
      if ( v14 )
      {
        result = (unsigned __int64)memcpy((void *)(*(_QWORD *)a1 + 4 * v6), a2, v14);
        LODWORD(v6) = *(_DWORD *)(a1 + 8);
      }
      v9 = 4 * (a3 >> 2) + 4 - a3;
      *(_DWORD *)(a1 + 8) = v13 + v6;
      if ( v9 != 2 )
        goto LABEL_10;
    }
    v10 = 0;
    goto LABEL_13;
  }
  return result;
}
