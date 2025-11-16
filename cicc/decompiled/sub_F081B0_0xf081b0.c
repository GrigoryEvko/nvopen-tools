// Function: sub_F081B0
// Address: 0xf081b0
//
void __fastcall sub_F081B0(__int64 *a1, __int64 a2, _QWORD *a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v8; // rdi
  __int64 v9; // rbx
  _QWORD *v10; // rax
  __int64 v11; // r13
  __int64 v12; // rax

  v6 = *(_QWORD *)(a2 + 16);
  if ( v6 )
  {
    while ( 1 )
    {
      v8 = *a1;
      v9 = *(_QWORD *)(v6 + 24);
      if ( *(_BYTE *)(*a1 + 28) )
      {
        v10 = *(_QWORD **)(v8 + 8);
        a4 = *(unsigned int *)(v8 + 20);
        a3 = &v10[a4];
        if ( v10 != a3 )
        {
          while ( v9 != *v10 )
          {
            if ( a3 == ++v10 )
              goto LABEL_14;
          }
          goto LABEL_7;
        }
LABEL_14:
        if ( (unsigned int)a4 < *(_DWORD *)(v8 + 16) )
        {
          *(_DWORD *)(v8 + 20) = a4 + 1;
          *a3 = v9;
          ++*(_QWORD *)v8;
          goto LABEL_10;
        }
      }
      sub_C8CC70(v8, *(_QWORD *)(v6 + 24), (__int64)a3, a4, a5, a6);
      if ( (_BYTE)a3 )
      {
LABEL_10:
        v11 = a1[1];
        v12 = *(unsigned int *)(v11 + 8);
        a4 = *(unsigned int *)(v11 + 12);
        if ( v12 + 1 > a4 )
        {
          sub_C8D5F0(a1[1], (const void *)(v11 + 16), v12 + 1, 8u, a5, a6);
          v12 = *(unsigned int *)(v11 + 8);
        }
        a3 = *(_QWORD **)v11;
        *(_QWORD *)(*(_QWORD *)v11 + 8 * v12) = v9;
        ++*(_DWORD *)(v11 + 8);
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          return;
      }
      else
      {
LABEL_7:
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          return;
      }
    }
  }
}
