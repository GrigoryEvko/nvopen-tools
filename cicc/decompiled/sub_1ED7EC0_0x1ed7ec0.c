// Function: sub_1ED7EC0
// Address: 0x1ed7ec0
//
void __fastcall sub_1ED7EC0(__int64 *a1, __int64 *a2, __int64 a3, char a4)
{
  __int64 v4; // rax
  unsigned int v5; // r12d
  __int64 v8; // rbx
  __int64 v9; // r12
  unsigned int v10; // eax
  int v11; // r8d
  int v12; // r9d
  __int64 v13; // rdx
  __int64 v14; // rax
  char v15; // cl
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 i; // rdx
  __int64 v19; // rax
  char v20; // si
  const void *v21; // [rsp+0h] [rbp-50h]
  __int64 v23; // [rsp+10h] [rbp-40h]

  v4 = *a1;
  v5 = *(_DWORD *)(*a1 + 72);
  if ( v5 )
  {
    v23 = v5;
    v8 = 0;
    v21 = (const void *)(a3 + 16);
    while ( 1 )
    {
      v9 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v4 + 64) + 8 * v8) + 8LL);
      v10 = *(_DWORD *)(a1[14] + 40 * v8);
      if ( v10 <= 2 )
        break;
      if ( v10 != 3 )
        goto LABEL_4;
      sub_1DC0B50(a1[5], *a2, v9, a3);
      v13 = a2[14] + 40LL * **(unsigned int **)(a1[14] + 40 * v8 + 24);
      v14 = (v9 >> 1) & 3;
      if ( !*(_BYTE *)(v13 + 32) || *(_DWORD *)v13 )
      {
        if ( !v14 )
          goto LABEL_4;
        if ( !a4 )
        {
LABEL_24:
          v19 = *(unsigned int *)(a3 + 8);
          if ( (unsigned int)v19 >= *(_DWORD *)(a3 + 12) )
          {
            sub_16CD150(a3, v21, 0, 8, v11, v12);
            v19 = *(unsigned int *)(a3 + 8);
          }
          *(_QWORD *)(*(_QWORD *)a3 + 8 * v19) = v9;
          ++*(_DWORD *)(a3 + 8);
          goto LABEL_4;
        }
        v15 = 0;
      }
      else
      {
        v15 = a4 & (v14 != 0);
        if ( !v15 )
          goto LABEL_4;
      }
      if ( (v9 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        BUG();
      v16 = *(_QWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 16);
      v17 = *(_QWORD *)(v16 + 32);
      for ( i = v17 + 40LL * *(unsigned int *)(v16 + 40); i != v17; v17 += 40 )
      {
        if ( !*(_BYTE *)v17 && (*(_BYTE *)(v17 + 3) & 0x10) != 0 && *((_DWORD *)a1 + 2) == *(_DWORD *)(v17 + 8) )
        {
          if ( (*(_DWORD *)v17 & 0xFFF00) != 0 )
          {
            v20 = *(_BYTE *)(v17 + 4);
            if ( (v20 & 1) != 0 && !v15 )
              *(_BYTE *)(v17 + 4) = v20 & 0xFE;
          }
          *(_BYTE *)(v17 + 3) &= ~0x40u;
        }
      }
      if ( !v15 )
        goto LABEL_24;
      if ( v23 == ++v8 )
        return;
LABEL_5:
      v4 = *a1;
    }
    if ( v10 && (unsigned __int8)sub_1ED7D20((__int64)a1, v8, (__int64)a2) )
      sub_1DC0B50(a1[5], *a1, v9, a3);
LABEL_4:
    if ( v23 == ++v8 )
      return;
    goto LABEL_5;
  }
}
