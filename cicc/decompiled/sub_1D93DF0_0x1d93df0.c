// Function: sub_1D93DF0
// Address: 0x1d93df0
//
void __fastcall sub_1D93DF0(char *a1, char *a2)
{
  char *v3; // r13
  __int64 v4; // rdi
  char *v5; // r15
  __int64 v6; // r14
  unsigned int v7; // edi
  int v8; // ecx
  __int64 v9; // rax
  unsigned int v10; // esi
  int v11; // edx
  __int64 *i; // rbx
  unsigned int v13; // edi
  int v14; // ecx
  __int64 v15; // rax
  unsigned int v16; // esi
  int v17; // edx
  __int64 v18; // rdi
  unsigned __int8 v19; // dl
  unsigned __int8 v20; // dl
  __int64 v21; // r8
  _QWORD *v22; // rbx
  __int64 v23; // r13
  __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // rdi

  if ( a1 != a2 && a2 != a1 + 8 )
  {
    v3 = a1 + 8;
    while ( 1 )
    {
      v6 = *(_QWORD *)v3;
      v7 = *(_DWORD *)(*(_QWORD *)v3 + 8LL);
      v8 = *(_DWORD *)(*(_QWORD *)v3 + 12LL);
      if ( v7 == 7 )
        v8 = -(*(_DWORD *)(v6 + 16) + v8);
      v9 = *(_QWORD *)a1;
      v10 = *(_DWORD *)(*(_QWORD *)a1 + 8LL);
      v11 = *(_DWORD *)(*(_QWORD *)a1 + 12LL);
      if ( v10 == 7 )
        v11 = -(*(_DWORD *)(v9 + 16) + v11);
      if ( v8 <= v11 )
      {
        if ( v8 != v11 )
          break;
        v20 = *(_BYTE *)(v6 + 20);
        if ( ((v20 & 1) != 0 || (*(_BYTE *)(v9 + 20) & 1) == 0)
          && (((*(_BYTE *)(v9 + 20) ^ v20) & 1) != 0
           || v7 >= v10
           && (v7 != v10
            || *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v6 + 16LL) + 48LL) >= *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v9 + 16LL)
                                                                                  + 48LL))) )
        {
          break;
        }
      }
      v5 = v3 + 8;
      *(_QWORD *)v3 = 0;
      v21 = v3 - a1;
      v22 = v3 + 8;
      v23 = (v3 - a1) >> 3;
      if ( v21 > 0 )
      {
        do
        {
          v24 = *(v22 - 2);
          v25 = *--v22;
          *(v22 - 1) = 0;
          *v22 = v24;
          if ( v25 )
            j_j___libc_free_0(v25, 24);
          --v23;
        }
        while ( v23 );
      }
      v26 = *(_QWORD *)a1;
      *(_QWORD *)a1 = v6;
      if ( v26 )
      {
        v3 = v5;
        j_j___libc_free_0(v26, 24);
        if ( a2 == v5 )
          return;
      }
      else
      {
LABEL_8:
        v3 = v5;
        if ( a2 == v5 )
          return;
      }
    }
    *(_QWORD *)v3 = 0;
    for ( i = (__int64 *)v3; ; --i )
    {
      v13 = *(_DWORD *)(v6 + 8);
      v14 = *(_DWORD *)(v6 + 12);
      if ( v13 == 7 )
        v14 = -(*(_DWORD *)(v6 + 16) + v14);
      v15 = *(i - 1);
      v16 = *(_DWORD *)(v15 + 8);
      v17 = *(_DWORD *)(v15 + 12);
      if ( v16 == 7 )
        v17 = -(*(_DWORD *)(v15 + 16) + v17);
      if ( v14 <= v17 )
      {
        if ( v14 != v17 )
          break;
        v19 = *(_BYTE *)(v6 + 20);
        if ( ((v19 & 1) != 0 || (*(_BYTE *)(v15 + 20) & 1) == 0)
          && (((*(_BYTE *)(v15 + 20) ^ v19) & 1) != 0
           || v13 >= v16
           && (v13 != v16
            || *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v6 + 16LL) + 48LL) >= *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v15 + 16LL)
                                                                                  + 48LL))) )
        {
          break;
        }
      }
      v18 = *i;
      *(i - 1) = 0;
      *i = v15;
      if ( v18 )
        j_j___libc_free_0(v18, 24);
    }
    v4 = *i;
    *i = v6;
    if ( v4 )
      j_j___libc_free_0(v4, 24);
    v5 = v3 + 8;
    goto LABEL_8;
  }
}
