// Function: sub_2754040
// Address: 0x2754040
//
void __fastcall sub_2754040(__int64 a1, __int64 a2, _QWORD *a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // r14
  __int64 v8; // rbx
  _QWORD *v9; // rax
  __int64 v10; // rax

  v6 = *(_QWORD *)(a1 + 16);
  if ( v6 )
  {
    v7 = (__int64)a3;
    while ( 1 )
    {
      v8 = *(_QWORD *)(v6 + 24);
      if ( *(_BYTE *)(v7 + 28) )
      {
        v9 = *(_QWORD **)(v7 + 8);
        a4 = *(unsigned int *)(v7 + 20);
        a3 = &v9[a4];
        if ( v9 != a3 )
        {
          while ( v8 != *v9 )
          {
            if ( a3 == ++v9 )
              goto LABEL_15;
          }
          goto LABEL_8;
        }
LABEL_15:
        if ( (unsigned int)a4 < *(_DWORD *)(v7 + 16) )
        {
          *(_DWORD *)(v7 + 20) = a4 + 1;
          *a3 = v8;
          ++*(_QWORD *)v7;
          goto LABEL_11;
        }
      }
      sub_C8CC70(v7, *(_QWORD *)(v6 + 24), (__int64)a3, a4, a5, a6);
      if ( (_BYTE)a3 )
      {
LABEL_11:
        v10 = *(unsigned int *)(a2 + 8);
        a4 = *(unsigned int *)(a2 + 12);
        if ( v10 + 1 > a4 )
        {
          sub_C8D5F0(a2, (const void *)(a2 + 16), v10 + 1, 8u, a5, a6);
          v10 = *(unsigned int *)(a2 + 8);
        }
        a3 = *(_QWORD **)a2;
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v10) = v8;
        ++*(_DWORD *)(a2 + 8);
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          return;
      }
      else
      {
LABEL_8:
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          return;
      }
    }
  }
}
