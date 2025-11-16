// Function: sub_16F69C0
// Address: 0x16f69c0
//
void __fastcall sub_16F69C0(unsigned int a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned int v10; // r13d
  char v11; // r12
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax

  if ( a1 <= 0x7FF )
  {
    v9 = *(unsigned int *)(a2 + 8);
    v10 = (a1 >> 6) | 0xFFFFFFC0;
    v11 = a1 & 0x3F | 0x80;
    if ( (unsigned int)v9 < *(_DWORD *)(a2 + 12) )
    {
LABEL_13:
      *(_BYTE *)(*(_QWORD *)a2 + v9) = v10;
      v12 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = v12;
      if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v12 )
      {
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 1, a5, a6);
        v12 = *(unsigned int *)(a2 + 8);
      }
      *(_BYTE *)(*(_QWORD *)a2 + v12) = v11;
      ++*(_DWORD *)(a2 + 8);
      return;
    }
LABEL_21:
    sub_16CD150(a2, (const void *)(a2 + 16), 0, 1, a5, a6);
    v9 = *(unsigned int *)(a2 + 8);
    goto LABEL_13;
  }
  if ( a1 > 0xFFFF )
  {
    if ( a1 > 0x10FFFF )
      return;
    v13 = *(unsigned int *)(a2 + 8);
    v11 = a1 & 0x3F | 0x80;
    v10 = (a1 >> 6) & 0x3F | 0xFFFFFF80;
    if ( (unsigned int)v13 >= *(_DWORD *)(a2 + 12) )
    {
      sub_16CD150(a2, (const void *)(a2 + 16), 0, 1, a5, a6);
      v13 = *(unsigned int *)(a2 + 8);
    }
    *(_BYTE *)(*(_QWORD *)a2 + v13) = (a1 >> 18) | 0xF0;
    v14 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v14;
    if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v14 )
    {
      sub_16CD150(a2, (const void *)(a2 + 16), 0, 1, a5, a6);
      v14 = *(unsigned int *)(a2 + 8);
    }
    *(_BYTE *)(*(_QWORD *)a2 + v14) = (a1 >> 12) & 0x3F | 0x80;
    v9 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v9;
    if ( *(_DWORD *)(a2 + 12) > (unsigned int)v9 )
      goto LABEL_13;
    goto LABEL_21;
  }
  v6 = *(unsigned int *)(a2 + 8);
  if ( (unsigned int)v6 >= *(_DWORD *)(a2 + 12) )
  {
    sub_16CD150(a2, (const void *)(a2 + 16), 0, 1, a5, a6);
    v6 = *(unsigned int *)(a2 + 8);
  }
  *(_BYTE *)(*(_QWORD *)a2 + v6) = (a1 >> 12) | 0xE0;
  v7 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
  *(_DWORD *)(a2 + 8) = v7;
  if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v7 )
  {
    sub_16CD150(a2, (const void *)(a2 + 16), 0, 1, a5, a6);
    v7 = *(unsigned int *)(a2 + 8);
  }
  *(_BYTE *)(*(_QWORD *)a2 + v7) = (a1 >> 6) & 0x3F | 0x80;
  v8 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
  *(_DWORD *)(a2 + 8) = v8;
  if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v8 )
  {
    sub_16CD150(a2, (const void *)(a2 + 16), 0, 1, a5, a6);
    v8 = *(unsigned int *)(a2 + 8);
  }
  *(_BYTE *)(*(_QWORD *)a2 + v8) = a1 & 0x3F | 0x80;
  ++*(_DWORD *)(a2 + 8);
}
