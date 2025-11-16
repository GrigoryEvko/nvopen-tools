// Function: sub_24FE290
// Address: 0x24fe290
//
void __fastcall sub_24FE290(__int64 *a1, __int64 a2, __int64 *a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // rdi
  __int64 *v9; // rax
  __int64 v10; // r12
  __int64 v11; // rax

  if ( a2 )
  {
    v7 = a2;
    while ( 1 )
    {
      v8 = *a1;
      if ( *(_BYTE *)(*a1 + 28) )
      {
        v9 = *(__int64 **)(v8 + 8);
        a4 = *(unsigned int *)(v8 + 20);
        a3 = &v9[a4];
        if ( v9 != a3 )
        {
          while ( *v9 != v7 )
          {
            if ( a3 == ++v9 )
              goto LABEL_15;
          }
          goto LABEL_8;
        }
LABEL_15:
        if ( (unsigned int)a4 < *(_DWORD *)(v8 + 16) )
        {
          *(_DWORD *)(v8 + 20) = a4 + 1;
          *a3 = v7;
          ++*(_QWORD *)v8;
          goto LABEL_11;
        }
      }
      sub_C8CC70(v8, v7, (__int64)a3, a4, a5, a6);
      if ( (_BYTE)a3 )
      {
LABEL_11:
        v10 = a1[1];
        v11 = *(unsigned int *)(v10 + 8);
        a4 = *(unsigned int *)(v10 + 12);
        if ( v11 + 1 > a4 )
        {
          sub_C8D5F0(a1[1], (const void *)(v10 + 16), v11 + 1, 8u, a5, a6);
          v11 = *(unsigned int *)(v10 + 8);
        }
        a3 = *(__int64 **)v10;
        *(_QWORD *)(*(_QWORD *)v10 + 8 * v11) = v7;
        ++*(_DWORD *)(v10 + 8);
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          return;
      }
      else
      {
LABEL_8:
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          return;
      }
    }
  }
}
