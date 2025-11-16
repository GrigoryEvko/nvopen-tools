// Function: sub_1EBE300
// Address: 0x1ebe300
//
__int64 __fastcall sub_1EBE300(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  char *v7; // rdx
  char *v8; // rdi
  __int64 v9; // r13
  int v10; // r14d
  unsigned __int64 v11; // rcx
  unsigned int v12; // eax
  __int64 v13; // r8
  __int64 v14; // r12
  unsigned int v15; // r12d
  __int64 v16; // rdx
  unsigned __int64 v18; // rcx
  char *v19; // rdx
  __int64 v20; // rax
  _QWORD *v21; // rsi
  _QWORD *v22; // rax
  __int64 v23; // rcx

  v7 = (char *)a1[111];
  v8 = (char *)a1[110];
  if ( v7 == v8 )
    return 0;
  v9 = a1[33];
  v10 = ~*((_DWORD *)v8 + 1);
  v11 = *(unsigned int *)(v9 + 408);
  v12 = v10 & 0x7FFFFFFF;
  v13 = 8LL * (v10 & 0x7FFFFFFF);
  if ( (v10 & 0x7FFFFFFFu) >= (unsigned int)v11 || (v14 = *(_QWORD *)(*(_QWORD *)(v9 + 400) + 8LL * v12)) == 0 )
  {
    v15 = v12 + 1;
    if ( (unsigned int)v11 < v12 + 1 )
    {
      v20 = v15;
      if ( v15 >= v11 )
      {
        if ( v15 > v11 )
        {
          if ( v15 > (unsigned __int64)*(unsigned int *)(v9 + 412) )
          {
            sub_16CD150(v9 + 400, (const void *)(v9 + 416), v15, 8, 8 * v10, a6);
            v11 = *(unsigned int *)(v9 + 408);
            v13 = 8LL * (v10 & 0x7FFFFFFF);
            v20 = v15;
          }
          v16 = *(_QWORD *)(v9 + 400);
          v21 = (_QWORD *)(v16 + 8 * v20);
          v22 = (_QWORD *)(v16 + 8 * v11);
          v23 = *(_QWORD *)(v9 + 416);
          if ( v21 != v22 )
          {
            do
              *v22++ = v23;
            while ( v21 != v22 );
            v16 = *(_QWORD *)(v9 + 400);
          }
          *(_DWORD *)(v9 + 408) = v15;
          goto LABEL_6;
        }
      }
      else
      {
        *(_DWORD *)(v9 + 408) = v15;
      }
    }
    v16 = *(_QWORD *)(v9 + 400);
LABEL_6:
    *(_QWORD *)(v16 + v13) = sub_1DBA290(v10);
    v14 = *(_QWORD *)(*(_QWORD *)(v9 + 400) + 8LL * (v10 & 0x7FFFFFFF));
    sub_1DBB110((_QWORD *)v9, v14);
    v7 = (char *)a1[111];
    v8 = (char *)a1[110];
  }
  if ( v7 - v8 > 8 )
  {
    v18 = *((_QWORD *)v7 - 1);
    v19 = v7 - 8;
    *(_DWORD *)v19 = *(_DWORD *)v8;
    *((_DWORD *)v19 + 1) = *((_DWORD *)v8 + 1);
    sub_1EBB620((__int64)v8, 0, (v19 - v8) >> 3, v18);
    v7 = (char *)a1[111];
  }
  a1[111] = v7 - 8;
  return v14;
}
