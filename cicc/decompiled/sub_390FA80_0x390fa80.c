// Function: sub_390FA80
// Address: 0x390fa80
//
void __fastcall sub_390FA80(unsigned int a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned int v8; // r13d
  __int64 v9; // rax
  __int64 v10; // rax

  if ( a1 <= 0x7F )
  {
    v6 = *(unsigned int *)(a2 + 8);
    if ( (unsigned int)v6 >= *(_DWORD *)(a2 + 12) )
    {
LABEL_9:
      sub_16CD150(a2, (const void *)(a2 + 16), 0, 1, a5, a6);
      v6 = *(unsigned int *)(a2 + 8);
    }
LABEL_6:
    *(_BYTE *)(*(_QWORD *)a2 + v6) = a1;
    ++*(_DWORD *)(a2 + 8);
    return;
  }
  if ( a1 <= 0x3FFF )
  {
    v7 = *(unsigned int *)(a2 + 8);
    v8 = (a1 >> 8) | 0xFFFFFF80;
    if ( (unsigned int)v7 < *(_DWORD *)(a2 + 12) )
      goto LABEL_8;
    goto LABEL_15;
  }
  if ( a1 <= 0x1FFFFFFF )
  {
    v9 = *(unsigned int *)(a2 + 8);
    if ( (unsigned int)v9 >= *(_DWORD *)(a2 + 12) )
    {
      sub_16CD150(a2, (const void *)(a2 + 16), 0, 1, a5, a6);
      v9 = *(unsigned int *)(a2 + 8);
    }
    *(_BYTE *)(*(_QWORD *)a2 + v9) = HIBYTE(a1) | 0xC0;
    v10 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v10;
    if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v10 )
    {
      sub_16CD150(a2, (const void *)(a2 + 16), 0, 1, a5, a6);
      v10 = *(unsigned int *)(a2 + 8);
    }
    *(_BYTE *)(*(_QWORD *)a2 + v10) = BYTE2(a1);
    v8 = a1 >> 8;
    v7 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v7;
    if ( *(_DWORD *)(a2 + 12) > (unsigned int)v7 )
    {
LABEL_8:
      *(_BYTE *)(*(_QWORD *)a2 + v7) = v8;
      v6 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = v6;
      if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v6 )
        goto LABEL_9;
      goto LABEL_6;
    }
LABEL_15:
    sub_16CD150(a2, (const void *)(a2 + 16), 0, 1, a5, a6);
    v7 = *(unsigned int *)(a2 + 8);
    goto LABEL_8;
  }
}
