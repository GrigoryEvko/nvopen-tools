// Function: sub_AA8FD0
// Address: 0xaa8fd0
//
__int64 __fastcall sub_AA8FD0(_QWORD *a1, __int64 a2)
{
  unsigned __int8 **v3; // r13
  __int64 v4; // rax
  unsigned __int8 **v5; // r12
  unsigned __int8 *v6; // rbx
  int v7; // eax
  __int64 v9; // rdi
  unsigned __int8 **v10; // rax
  __int64 v11; // rcx
  unsigned __int8 **v12; // rdx
  __int64 v13; // r15
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rcx
  int v16; // eax
  unsigned __int8 **v17; // rdx
  char v18; // dl

  v3 = (unsigned __int8 **)a2;
  v4 = 4LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v5 = *(unsigned __int8 ***)(a2 - 8);
    v3 = &v5[v4];
  }
  else
  {
    v5 = (unsigned __int8 **)(a2 - v4 * 8);
  }
  if ( v5 != v3 )
  {
    while ( 1 )
    {
      v6 = *v5;
      v7 = **v5;
      if ( (unsigned int)(v7 - 12) <= 1 )
        goto LABEL_5;
      if ( (unsigned int)(v7 - 9) > 2 )
        return 0;
      v9 = *a1;
      if ( *(_BYTE *)(*a1 + 28LL) )
      {
        v10 = *(unsigned __int8 ***)(v9 + 8);
        v11 = *(unsigned int *)(v9 + 20);
        v12 = &v10[v11];
        if ( v10 != v12 )
        {
          while ( v6 != *v10 )
          {
            if ( v12 == ++v10 )
              goto LABEL_13;
          }
          goto LABEL_5;
        }
LABEL_13:
        if ( (unsigned int)v11 < *(_DWORD *)(v9 + 16) )
        {
          *(_DWORD *)(v9 + 20) = v11 + 1;
          *v12 = v6;
          ++*(_QWORD *)v9;
          goto LABEL_15;
        }
      }
      sub_C8CC70(v9, *v5);
      if ( v18 )
      {
LABEL_15:
        v13 = a1[1];
        v14 = *(unsigned int *)(v13 + 8);
        v15 = *(unsigned int *)(v13 + 12);
        v16 = *(_DWORD *)(v13 + 8);
        if ( v14 >= v15 )
        {
          if ( v15 < v14 + 1 )
          {
            sub_C8D5F0(a1[1], v13 + 16, v14 + 1, 8);
            v14 = *(unsigned int *)(v13 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v13 + 8 * v14) = v6;
          ++*(_DWORD *)(v13 + 8);
          goto LABEL_5;
        }
        v17 = (unsigned __int8 **)(*(_QWORD *)v13 + 8 * v14);
        if ( v17 )
        {
          *v17 = v6;
          v16 = *(_DWORD *)(v13 + 8);
        }
        v5 += 4;
        *(_DWORD *)(v13 + 8) = v16 + 1;
        if ( v3 == v5 )
          return 1;
      }
      else
      {
LABEL_5:
        v5 += 4;
        if ( v3 == v5 )
          return 1;
      }
    }
  }
  return 1;
}
