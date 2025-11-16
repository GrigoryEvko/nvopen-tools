// Function: sub_20FD7B0
// Address: 0x20fd7b0
//
void __fastcall sub_20FD7B0(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v9; // rax
  int v10; // r9d
  __int64 *v11; // rsi
  unsigned int v12; // eax
  __int64 v13; // r8
  __int64 v14; // rdi
  __int64 *v15; // rcx
  __int64 *v16; // rdx
  __int64 v17; // r8
  __int64 v18; // r8
  __int64 v19; // rcx
  __int64 v20; // r9
  _DWORD *v21; // rax
  unsigned __int64 *v22; // rcx
  __int64 *v23; // rax
  __int64 v24; // rdx
  int v25; // ecx
  _DWORD *v26; // rdi
  unsigned int v27; // edx
  unsigned int *v28; // rax

  v7 = *(_QWORD *)(a1 + 8);
  v8 = *(_QWORD *)a1;
  v9 = v7 + 16LL * *(unsigned int *)(a1 + 16) - 16;
  v10 = *(_DWORD *)(v9 + 8);
  v11 = *(__int64 **)v9;
  if ( v10 == 1 )
  {
    v23 = *(__int64 **)(v8 + 200);
    v24 = *v23;
    *v11 = *v23;
    *v23 = (__int64)v11;
    sub_20FD590(a1, *(_DWORD *)(v8 + 192), v24, a4, a5, 1);
    if ( a2 )
    {
      if ( *(_DWORD *)(v8 + 192) )
      {
        v25 = *(_DWORD *)(a1 + 16);
        if ( v25 )
        {
          v26 = *(_DWORD **)(a1 + 8);
          v27 = v26[3];
          if ( v27 < v26[2] )
          {
            v28 = v26 + 7;
            while ( !v27 )
            {
              if ( v28 == &v26[4 * (v25 - 1) + 7] )
              {
                *(_QWORD *)v8 = **(_QWORD **)&v26[4 * v25 - 4];
                return;
              }
              v27 = *v28;
              v28 += 4;
            }
          }
        }
      }
    }
  }
  else
  {
    v12 = *(_DWORD *)(v9 + 12) + 1;
    if ( v10 != v12 )
    {
      do
      {
        v13 = v12;
        v14 = v12++ - 1;
        v15 = &v11[2 * v13];
        v16 = &v11[2 * v14];
        *v16 = *v15;
        v16[1] = v15[1];
        v11[v14 + 16] = v11[v13 + 16];
      }
      while ( v10 != v12 );
      v7 = *(_QWORD *)(a1 + 8);
      v12 = *(_DWORD *)(v7 + 16LL * *(unsigned int *)(a1 + 16) - 8);
    }
    v17 = *(unsigned int *)(v8 + 192);
    *(_DWORD *)(v7 + 16 * v17 + 8) = v12 - 1;
    if ( (_DWORD)v17 )
    {
      v22 = (unsigned __int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 16LL * (unsigned int)(v17 - 1))
                               + 8LL * *(unsigned int *)(*(_QWORD *)(a1 + 8) + 16LL * (unsigned int)(v17 - 1) + 12));
      *v22 = (v12 - 2) | *v22 & 0xFFFFFFFFFFFFFFC0LL;
    }
    v18 = *(_QWORD *)(a1 + 8);
    v19 = *(unsigned int *)(a1 + 16);
    v20 = v18 + 16 * v19 - 16;
    if ( *(_DWORD *)(v20 + 12) == v12 - 1 )
    {
      sub_20FCF40(a1, *(_DWORD *)(v8 + 192), v11[2 * v12 - 3]);
      sub_39460A0(a1 + 8, *(unsigned int *)(v8 + 192));
    }
    else if ( a2 )
    {
      if ( (_DWORD)v19 )
      {
        v21 = (_DWORD *)(v18 + 12);
        while ( !*v21 )
        {
          v21 += 4;
          if ( (_DWORD *)(v18 + 16LL * (unsigned int)(v19 - 1) + 28) == v21 )
            goto LABEL_24;
        }
      }
      else
      {
LABEL_24:
        *(_QWORD *)v8 = **(_QWORD **)v20;
      }
    }
  }
}
