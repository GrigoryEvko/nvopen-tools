// Function: sub_38B8770
// Address: 0x38b8770
//
__int64 __fastcall sub_38B8770(__int64 a1, unsigned __int8 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 v7; // r13
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rcx
  unsigned int v11; // esi
  char *v12; // rdx
  char *v13; // rdi
  __int64 v15; // rax

  v7 = a4 & 0xFFFFFFFFFFFFFFFBLL | (4LL * a2);
  v9 = *(_QWORD *)(a1 + 280);
  v10 = a4 & 0xFFFFFFFFFFFFFFFBLL | (4LL * (a2 == 0));
  if ( *(_QWORD *)(a1 + 272) >= v9 )
    v9 = *(_QWORD *)(a1 + 272);
  v11 = *(_DWORD *)(a1 + 8);
  v12 = (char *)(*(_QWORD *)a1 + 16LL * v11);
  v13 = (char *)(*(_QWORD *)a1 + 16 * v9);
  if ( v12 == v13 )
  {
LABEL_6:
    if ( v11 >= *(_DWORD *)(a1 + 12) )
    {
      sub_16CD150(a1, (const void *)(a1 + 16), 0, 16, a5, a6);
      v12 = (char *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
    }
    *(_QWORD *)v12 = a3;
    *((_QWORD *)v12 + 1) = v7;
    ++*(_DWORD *)(a1 + 8);
    return 1;
  }
  else
  {
    while ( 1 )
    {
      if ( a3 == *(_QWORD *)v13 )
      {
        v15 = *((_QWORD *)v13 + 1);
        if ( v7 == v15 )
          return 0;
        if ( v10 == v15 )
          break;
      }
      v13 += 16;
      if ( v12 == v13 )
        goto LABEL_6;
    }
    if ( v12 != v13 + 16 )
    {
      memmove(v13, v13 + 16, v12 - (v13 + 16));
      v11 = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v11 - 1;
    return 0;
  }
}
