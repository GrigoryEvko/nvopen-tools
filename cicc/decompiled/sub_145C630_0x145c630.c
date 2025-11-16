// Function: sub_145C630
// Address: 0x145c630
//
__int64 __fastcall sub_145C630(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rdi
  _QWORD *v6; // rax
  __int64 v7; // rsi
  char v8; // dl
  char v9; // cl
  _QWORD *v10; // rdx
  _QWORD *v11; // rdx
  _QWORD *v13; // r8
  unsigned int v14; // r9d
  _QWORD *v15; // rcx
  __int64 v16; // rax
  int v17; // eax

  v5 = *(_QWORD **)(a2 + 16);
  v6 = *(_QWORD **)(a2 + 8);
  if ( v5 != v6 )
    goto LABEL_2;
  v13 = &v5[*(unsigned int *)(a2 + 28)];
  v14 = *(_DWORD *)(a2 + 28);
  if ( v5 == v13 )
    goto LABEL_19;
  v15 = 0;
  do
  {
    if ( a3 == *v6 )
    {
      v7 = *(_QWORD *)a2;
      v9 = 0;
      goto LABEL_16;
    }
    if ( *v6 == -2 )
      v15 = v6;
    ++v6;
  }
  while ( v13 != v6 );
  if ( !v15 )
  {
LABEL_19:
    if ( v14 >= *(_DWORD *)(a2 + 24) )
    {
LABEL_2:
      v6 = (_QWORD *)sub_16CCBA0(a2, a3);
      v5 = *(_QWORD **)(a2 + 16);
      v7 = *(_QWORD *)a2;
      v9 = v8;
      v10 = *(_QWORD **)(a2 + 8);
      goto LABEL_3;
    }
    v9 = 1;
    *(_DWORD *)(a2 + 28) = v14 + 1;
    *v13 = a3;
    v10 = *(_QWORD **)(a2 + 8);
    v5 = *(_QWORD **)(a2 + 16);
    v7 = *(_QWORD *)a2 + 1LL;
    v17 = *(_DWORD *)(a2 + 28);
    *(_QWORD *)a2 = v7;
    v6 = &v10[v17 - 1];
  }
  else
  {
    *v15 = a3;
    v16 = *(_QWORD *)a2;
    --*(_DWORD *)(a2 + 32);
    v5 = *(_QWORD **)(a2 + 16);
    v7 = v16 + 1;
    v10 = *(_QWORD **)(a2 + 8);
    v6 = v15;
    v9 = 1;
    *(_QWORD *)a2 = v7;
  }
LABEL_3:
  if ( v5 == v10 )
LABEL_16:
    v11 = &v5[*(unsigned int *)(a2 + 28)];
  else
    v11 = &v5[*(unsigned int *)(a2 + 24)];
  while ( v11 != v6 && *v6 >= 0xFFFFFFFFFFFFFFFELL )
    ++v6;
  *(_QWORD *)a1 = v6;
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 8) = v11;
  *(_QWORD *)(a1 + 24) = v7;
  *(_BYTE *)(a1 + 32) = v9;
  return a1;
}
