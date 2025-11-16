// Function: sub_1DBB1A0
// Address: 0x1dbb1a0
//
void __fastcall sub_1DBB1A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // rcx
  int v7; // r13d
  int v9; // ebx
  __int64 v10; // rcx
  __int64 v11; // r15
  __int64 v12; // rax
  unsigned int v13; // r14d
  unsigned __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // rdi
  _QWORD *v17; // rsi
  _QWORD *v18; // rdx
  const void *v19; // [rsp+8h] [rbp-38h]

  v6 = *(_QWORD *)(a1 + 240);
  v7 = *(_DWORD *)(v6 + 32);
  if ( v7 )
  {
    v9 = 0;
    v19 = (const void *)(a1 + 416);
    while ( 1 )
    {
      v11 = v9 & 0x7FFFFFFF;
      v12 = *(_QWORD *)(*(_QWORD *)(v6 + 24) + 16 * v11 + 8);
      if ( !v12 )
        goto LABEL_5;
      if ( (*(_BYTE *)(v12 + 4) & 8) == 0 )
        break;
      while ( 1 )
      {
        v12 = *(_QWORD *)(v12 + 32);
        if ( !v12 )
          break;
        if ( (*(_BYTE *)(v12 + 4) & 8) == 0 )
          goto LABEL_9;
      }
      if ( v7 == ++v9 )
        return;
LABEL_6:
      v6 = *(_QWORD *)(a1 + 240);
    }
LABEL_9:
    v13 = (v9 & 0x7FFFFFFF) + 1;
    v14 = *(unsigned int *)(a1 + 408);
    if ( v13 > (unsigned int)v14 )
    {
      v15 = v13;
      if ( v13 >= v14 )
      {
        if ( v13 > v14 )
        {
          if ( v13 > (unsigned __int64)*(unsigned int *)(a1 + 412) )
          {
            sub_16CD150(a1 + 400, v19, v13, 8, v13, a6);
            v14 = *(unsigned int *)(a1 + 408);
            v15 = v13;
          }
          v10 = *(_QWORD *)(a1 + 400);
          v16 = *(_QWORD *)(a1 + 416);
          v17 = (_QWORD *)(v10 + 8 * v15);
          v18 = (_QWORD *)(v10 + 8 * v14);
          if ( v17 != v18 )
          {
            do
              *v18++ = v16;
            while ( v17 != v18 );
            v10 = *(_QWORD *)(a1 + 400);
          }
          *(_DWORD *)(a1 + 408) = v13;
          goto LABEL_4;
        }
      }
      else
      {
        *(_DWORD *)(a1 + 408) = v13;
      }
    }
    v10 = *(_QWORD *)(a1 + 400);
LABEL_4:
    *(_QWORD *)(v10 + 8 * v11) = sub_1DBA290(v9 | 0x80000000);
    sub_1DBB110((_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 400) + 8 * v11));
LABEL_5:
    if ( v7 == ++v9 )
      return;
    goto LABEL_6;
  }
}
