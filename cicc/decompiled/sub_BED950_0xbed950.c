// Function: sub_BED950
// Address: 0xbed950
//
__int64 __fastcall sub_BED950(__int64 a1, __int64 a2, __int64 a3)
{
  char v6; // cl
  _QWORD *v7; // rdi
  _QWORD *v8; // rdx
  unsigned int v9; // r8d
  _QWORD *v10; // rax
  __int64 v11; // rsi
  _QWORD *v12; // rdx
  char v14; // dl
  char v15; // dl
  int v16; // eax

  v6 = *(_BYTE *)(a2 + 28);
  if ( !v6 )
    goto LABEL_10;
  v7 = *(_QWORD **)(a2 + 8);
  v8 = &v7[*(unsigned int *)(a2 + 20)];
  v9 = *(_DWORD *)(a2 + 20);
  if ( v7 == v8 )
  {
LABEL_9:
    if ( v9 < *(_DWORD *)(a2 + 16) )
    {
      *(_DWORD *)(a2 + 20) = v9 + 1;
      *v8 = a3;
      v7 = *(_QWORD **)(a2 + 8);
      v15 = *(_BYTE *)(a2 + 28);
      v11 = *(_QWORD *)a2 + 1LL;
      v16 = *(_DWORD *)(a2 + 20);
      *(_QWORD *)a2 = v11;
      v10 = &v7[v16 - 1];
LABEL_11:
      if ( !v15 )
      {
        v12 = &v7[*(unsigned int *)(a2 + 16)];
        goto LABEL_15;
      }
      goto LABEL_7;
    }
LABEL_10:
    v10 = (_QWORD *)sub_C8CC70(a2, a3);
    v11 = *(_QWORD *)a2;
    v7 = *(_QWORD **)(a2 + 8);
    v6 = v14;
    v15 = *(_BYTE *)(a2 + 28);
    goto LABEL_11;
  }
  v10 = *(_QWORD **)(a2 + 8);
  while ( a3 != *v10 )
  {
    if ( v8 == ++v10 )
      goto LABEL_9;
  }
  v11 = *(_QWORD *)a2;
  v6 = 0;
LABEL_7:
  v12 = &v7[*(unsigned int *)(a2 + 20)];
  while ( v10 != v12 )
  {
    if ( *v10 < 0xFFFFFFFFFFFFFFFELL )
      break;
    ++v10;
LABEL_15:
    ;
  }
  *(_QWORD *)a1 = v10;
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 8) = v12;
  *(_QWORD *)(a1 + 24) = v11;
  *(_BYTE *)(a1 + 32) = v6;
  return a1;
}
