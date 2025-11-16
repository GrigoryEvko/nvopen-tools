// Function: sub_2BB7BD0
// Address: 0x2bb7bd0
//
void __fastcall sub_2BB7BD0(__int64 a1, unsigned __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  int v12; // r14d
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rbx
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rdx
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rsi
  __int64 v24; // rcx

  if ( (unsigned __int64 *)a1 != a2 )
  {
    v8 = *a2;
    v9 = (unsigned __int64)(a2 + 2);
    if ( (unsigned __int64 *)*a2 == a2 + 2 )
    {
      v10 = *((unsigned int *)a2 + 2);
      v11 = *(unsigned int *)(a1 + 8);
      v12 = v10;
      if ( v10 <= v11 )
      {
        if ( v10 )
        {
          v19 = *(_QWORD *)a1;
          v20 = *(_QWORD *)a1 + 16 * v10;
          do
          {
            v21 = *(_QWORD *)(v9 + 8);
            v19 += 16LL;
            v9 += 16LL;
            *(_QWORD *)(v19 - 8) = v21;
            *(_DWORD *)(v19 - 12) = *(_DWORD *)(v9 - 12);
            *(_DWORD *)(v19 - 16) = *(_DWORD *)(v9 - 16);
          }
          while ( v20 != v19 );
        }
      }
      else
      {
        v13 = *(unsigned int *)(a1 + 12);
        if ( v10 > v13 )
        {
          *(_DWORD *)(a1 + 8) = 0;
          sub_2BB7B30(a1, v10, v13, a4, a5, a6);
          v9 = *a2;
          v10 = *((unsigned int *)a2 + 2);
          v11 = 0;
          v14 = *a2;
        }
        else
        {
          v14 = v9;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v22 = *(_QWORD *)a1;
            v11 *= 16LL;
            v23 = *(_QWORD *)a1 + v11;
            do
            {
              v24 = *(_QWORD *)(v9 + 8);
              v22 += 16LL;
              v9 += 16LL;
              *(_QWORD *)(v22 - 8) = v24;
              *(_DWORD *)(v22 - 12) = *(_DWORD *)(v9 - 12);
              *(_DWORD *)(v22 - 16) = *(_DWORD *)(v9 - 16);
            }
            while ( v22 != v23 );
            v9 = *a2;
            v10 = *((unsigned int *)a2 + 2);
            v14 = *a2 + v11;
          }
        }
        v15 = *(_QWORD *)a1 + v11;
        v16 = 16 * v10 + v9;
        v17 = v15 + v16 - v14;
        if ( v16 != v14 )
        {
          do
          {
            if ( v15 )
            {
              *(_DWORD *)v15 = *(_DWORD *)v14;
              *(_DWORD *)(v15 + 4) = *(_DWORD *)(v14 + 4);
              *(_QWORD *)(v15 + 8) = *(_QWORD *)(v14 + 8);
            }
            v15 += 16LL;
            v14 += 16LL;
          }
          while ( v17 != v15 );
        }
      }
      *(_DWORD *)(a1 + 8) = v12;
      *((_DWORD *)a2 + 2) = 0;
    }
    else
    {
      v18 = *(_QWORD *)a1;
      if ( v18 != a1 + 16 )
      {
        _libc_free(v18);
        v8 = *a2;
      }
      *(_QWORD *)a1 = v8;
      *(_DWORD *)(a1 + 8) = *((_DWORD *)a2 + 2);
      *(_DWORD *)(a1 + 12) = *((_DWORD *)a2 + 3);
      *a2 = v9;
      a2[1] = 0;
    }
  }
}
