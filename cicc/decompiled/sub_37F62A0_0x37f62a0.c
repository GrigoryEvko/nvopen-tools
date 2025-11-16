// Function: sub_37F62A0
// Address: 0x37f62a0
//
void __fastcall sub_37F62A0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 *v12; // rbx
  __int64 *v13; // r14
  __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 *v16; // rax

  v6 = sub_37F6040(a1, a2, a3);
  if ( !v6 )
  {
    v11 = *(_QWORD *)(a2 + 24);
    v12 = *(__int64 **)(v11 + 64);
    v13 = &v12[*(unsigned int *)(v11 + 72)];
    while ( v13 != v12 )
    {
      v14 = *v12++;
      sub_37F5FF0(a1, v14, a3, a4, v9, v10);
    }
    return;
  }
  v15 = v6;
  if ( !*(_BYTE *)(a4 + 28) )
  {
LABEL_11:
    sub_C8CC70(a4, v15, (__int64)v7, v8, v9, v10);
    return;
  }
  v16 = *(__int64 **)(a4 + 8);
  v8 = *(unsigned int *)(a4 + 20);
  v7 = &v16[v8];
  if ( v16 == v7 )
  {
LABEL_12:
    if ( (unsigned int)v8 < *(_DWORD *)(a4 + 16) )
    {
      *(_DWORD *)(a4 + 20) = v8 + 1;
      *v7 = v15;
      ++*(_QWORD *)a4;
      return;
    }
    goto LABEL_11;
  }
  while ( v15 != *v16 )
  {
    if ( v7 == ++v16 )
      goto LABEL_12;
  }
}
