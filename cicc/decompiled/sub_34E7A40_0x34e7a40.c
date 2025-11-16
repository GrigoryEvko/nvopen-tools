// Function: sub_34E7A40
// Address: 0x34e7a40
//
void __fastcall sub_34E7A40(unsigned __int64 *a1, unsigned __int64 *a2)
{
  unsigned __int64 *v3; // r13
  unsigned __int64 v4; // rdi
  unsigned __int64 *v5; // r15
  unsigned __int64 v6; // r14
  unsigned int v7; // edi
  int v8; // ecx
  unsigned __int64 v9; // rax
  unsigned int v10; // esi
  int v11; // edx
  unsigned __int64 *i; // rbx
  unsigned int v13; // edi
  int v14; // ecx
  unsigned __int64 v15; // rax
  unsigned int v16; // esi
  int v17; // edx
  unsigned __int64 v18; // rdi
  unsigned __int8 v19; // dl
  unsigned __int8 v20; // dl
  __int64 v21; // r8
  _QWORD *v22; // rbx
  __int64 v23; // r13
  __int64 v24; // rdx
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi

  if ( a1 != a2 && a2 != a1 + 1 )
  {
    v3 = a1 + 1;
    while ( 1 )
    {
      v6 = *v3;
      v7 = *(_DWORD *)(*v3 + 8);
      v8 = *(_DWORD *)(*v3 + 12);
      if ( v7 == 7 )
        v8 = -(*(_DWORD *)(v6 + 16) + v8);
      v9 = *a1;
      v10 = *(_DWORD *)(*a1 + 8);
      v11 = *(_DWORD *)(*a1 + 12);
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
            || *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v6 + 16LL) + 24LL) >= *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v9 + 16LL)
                                                                                  + 24LL))) )
        {
          break;
        }
      }
      v5 = v3 + 1;
      *v3 = 0;
      v21 = (char *)v3 - (char *)a1;
      v22 = v3 + 1;
      v23 = v3 - a1;
      if ( v21 > 0 )
      {
        do
        {
          v24 = *(v22 - 2);
          v25 = *--v22;
          *(v22 - 1) = 0;
          *v22 = v24;
          if ( v25 )
            j_j___libc_free_0(v25);
          --v23;
        }
        while ( v23 );
      }
      v26 = *a1;
      *a1 = v6;
      if ( v26 )
      {
        v3 = v5;
        j_j___libc_free_0(v26);
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
    *v3 = 0;
    for ( i = v3; ; --i )
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
            || *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v6 + 16LL) + 24LL) >= *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v15 + 16LL)
                                                                                  + 24LL))) )
        {
          break;
        }
      }
      v18 = *i;
      *(i - 1) = 0;
      *i = v15;
      if ( v18 )
        j_j___libc_free_0(v18);
    }
    v4 = *i;
    *i = v6;
    if ( v4 )
      j_j___libc_free_0(v4);
    v5 = v3 + 1;
    goto LABEL_8;
  }
}
