// Function: sub_2425BF0
// Address: 0x2425bf0
//
void __fastcall sub_2425BF0(unsigned __int64 *a1, unsigned __int64 *a2)
{
  unsigned __int64 *v2; // rbx
  unsigned __int64 v4; // r14
  unsigned int v5; // ecx
  bool v6; // cf
  unsigned __int64 *v7; // r12
  __int64 v8; // r15
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rax
  unsigned int v13; // edx
  bool v14; // cf
  unsigned __int64 v15; // rdi
  unsigned int v16; // edx

  if ( a1 != a2 )
  {
    v2 = a1 + 1;
    if ( a2 != a1 + 1 )
    {
      while ( 1 )
      {
        v4 = *v2;
        v5 = *(_DWORD *)(*a1 + 32);
        v6 = *(_DWORD *)(*v2 + 32) < v5;
        if ( *(_DWORD *)(*v2 + 32) == v5 )
          v6 = *(_DWORD *)(v4 + 36) < *(_DWORD *)(*a1 + 36);
        v7 = v2 + 1;
        if ( v6 )
        {
          *v2 = 0;
          v8 = v2 - a1;
          if ( (char *)v2 - (char *)a1 > 0 )
          {
            do
            {
              v9 = *(v2 - 1);
              v10 = *v2--;
              *v2 = 0;
              v2[1] = v9;
              if ( v10 )
                j_j___libc_free_0(v10);
              --v8;
            }
            while ( v8 );
          }
          v11 = *a1;
          *a1 = v4;
          if ( v11 )
LABEL_11:
            j_j___libc_free_0(v11);
          if ( a2 == v7 )
            return;
          goto LABEL_13;
        }
        *v2 = 0;
        v12 = *(v2 - 1);
        v13 = *(_DWORD *)(v12 + 32);
        v14 = *(_DWORD *)(v4 + 32) < v13;
        if ( *(_DWORD *)(v4 + 32) != v13 )
          break;
        while ( *(_DWORD *)(v4 + 36) < *(_DWORD *)(v12 + 36) )
        {
LABEL_16:
          v15 = *v2;
          *(v2 - 1) = 0;
          *v2 = v12;
          if ( v15 )
            j_j___libc_free_0(v15);
          v12 = *(v2 - 2);
          --v2;
          v16 = *(_DWORD *)(v12 + 32);
          v14 = *(_DWORD *)(v4 + 32) < v16;
          if ( *(_DWORD *)(v4 + 32) != v16 )
            goto LABEL_15;
        }
LABEL_20:
        v11 = *v2;
        *v2 = v4;
        if ( v11 )
          goto LABEL_11;
        if ( a2 == v7 )
          return;
LABEL_13:
        v2 = v7;
      }
LABEL_15:
      if ( !v14 )
        goto LABEL_20;
      goto LABEL_16;
    }
  }
}
