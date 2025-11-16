// Function: sub_2721330
// Address: 0x2721330
//
void __fastcall sub_2721330(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v8; // rax
  _QWORD *v9; // rbx
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  int v12; // r14d
  unsigned __int64 v13; // rdx
  _QWORD *v14; // rdx
  _QWORD *v15; // rax
  _QWORD *i; // rsi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rsi

  if ( a1 != a2 )
  {
    v8 = *(_QWORD **)a2;
    v9 = (_QWORD *)(a2 + 16);
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v10 = *(unsigned int *)(a2 + 8);
      v11 = *(unsigned int *)(a1 + 8);
      v12 = v10;
      if ( v10 <= v11 )
      {
        if ( v10 )
        {
          v18 = *(_QWORD *)a1;
          do
          {
            v19 = v9[2];
            v9 += 3;
            v18 += 24LL;
            *(_QWORD *)(v18 - 8) = v19;
            *(_QWORD *)(v18 - 16) = *(v9 - 2);
            *(_QWORD *)(v18 - 24) = *(v9 - 3);
          }
          while ( v9 != (_QWORD *)(a2 + 24 * v10 + 16) );
        }
      }
      else
      {
        v13 = *(unsigned int *)(a1 + 12);
        if ( v10 > v13 )
        {
          *(_DWORD *)(a1 + 8) = 0;
          sub_2721290(a1, v10, v13, a4, a5, a6);
          v9 = *(_QWORD **)a2;
          v10 = *(unsigned int *)(a2 + 8);
          v11 = 0;
          v14 = *(_QWORD **)a2;
        }
        else
        {
          v14 = v9;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v20 = *(_QWORD *)a1;
            v21 = 24 * v11;
            v11 *= 24LL;
            do
            {
              v22 = v9[2];
              v9 += 3;
              v20 += 24LL;
              *(_QWORD *)(v20 - 8) = v22;
              *(_QWORD *)(v20 - 16) = *(v9 - 2);
              *(_QWORD *)(v20 - 24) = *(v9 - 3);
            }
            while ( v9 != (_QWORD *)(a2 + v21 + 16) );
            v9 = *(_QWORD **)a2;
            v10 = *(unsigned int *)(a2 + 8);
            v14 = (_QWORD *)(*(_QWORD *)a2 + v21);
          }
        }
        v15 = (_QWORD *)(*(_QWORD *)a1 + v11);
        for ( i = &v9[3 * v10]; i != v14; v15 += 3 )
        {
          if ( v15 )
          {
            *v15 = *v14;
            v15[1] = v14[1];
            v15[2] = v14[2];
          }
          v14 += 3;
        }
      }
      *(_DWORD *)(a1 + 8) = v12;
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v17 = *(_QWORD *)a1;
      if ( v17 != a1 + 16 )
      {
        _libc_free(v17);
        v8 = *(_QWORD **)a2;
      }
      *(_QWORD *)a1 = v8;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v9;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
